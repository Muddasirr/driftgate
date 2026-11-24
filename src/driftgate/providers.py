"""Model providers.

`OpenAIProvider` speaks the OpenAI chat-completions wire format, which OpenAI,
Groq, Together, Fireworks, vLLM and Ollama all accept. Claude models use
`driftgate.anthropic_provider`, built on the official Anthropic SDK. Cases are
always written in one (OpenAI-style) format and each provider translates on the
way out.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar, Protocol

import httpx

from driftgate.cache import ResponseCache
from driftgate.config import (
    AnthropicProviderSpec,
    MockProviderSpec,
    OpenAIProviderSpec,
    Pricing,
    ToolCall,
)
from driftgate.results import Completion

RETRYABLE_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504})


class ProviderError(Exception):
    """A call failed in a way that retrying won't fix (or retries ran out)."""


@dataclass(frozen=True)
class Request:
    case_id: str
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] | None = None
    temperature: float = 0.0
    max_tokens: int = 512
    attempt: int = 0


class Provider(Protocol):
    id: str

    async def complete(self, request: Request) -> Completion: ...

    async def aclose(self) -> None: ...


def cost_of(pricing: Pricing | None, input_tokens: int, output_tokens: int) -> float | None:
    if pricing is None:
        return None
    return (input_tokens * pricing.input_per_mtok + output_tokens * pricing.output_per_mtok) / 1e6


class HTTPProvider(ABC):
    """Shared request loop: cache lookup, POST with retries and backoff, parse."""

    retryable_status: ClassVar[frozenset[int]] = RETRYABLE_STATUS

    def __init__(
        self,
        spec: OpenAIProviderSpec,
        *,
        cache: ResponseCache | None = None,
        client: httpx.AsyncClient | None = None,
        sleep: Any = asyncio.sleep,
    ) -> None:
        self.id = spec.id
        self.spec = spec
        self._cache = cache
        self._sleep = sleep
        self._client = client or httpx.AsyncClient(timeout=spec.timeout_s)

    @property
    @abstractmethod
    def url(self) -> str: ...

    @property
    @abstractmethod
    def headers(self) -> dict[str, str]: ...

    @abstractmethod
    def payload(self, request: Request) -> dict[str, Any]: ...

    @abstractmethod
    def parse(self, body: dict[str, Any], latency_ms: float) -> Completion: ...

    async def aclose(self) -> None:
        await self._client.aclose()

    async def complete(self, request: Request) -> Completion:
        payload = self.payload(request)
        cache_key = None
        if self._cache is not None:
            cache_key = self._cache.key({"url": self.url, "attempt": request.attempt, **payload})
            hit = self._cache.get(cache_key)
            if hit is not None:
                return hit.model_copy(update={"cached": True})

        completion = await self._post_with_retries(payload)
        if self._cache is not None and cache_key is not None:
            self._cache.put(cache_key, completion)
        return completion

    async def _post_with_retries(self, payload: dict[str, Any]) -> Completion:
        last_error = "unknown error"
        for attempt in range(self.spec.max_retries + 1):
            started = time.perf_counter()
            try:
                response = await self._client.post(self.url, json=payload, headers=self.headers)
            except (httpx.TransportError, httpx.TimeoutException) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                await self._backoff(attempt, None)
                continue
            latency_ms = (time.perf_counter() - started) * 1000

            if response.status_code in self.retryable_status:
                last_error = f"HTTP {response.status_code}: {_snippet(response)}"
                await self._backoff(attempt, response.headers.get("retry-after"))
                continue
            if response.status_code >= 400:
                raise ProviderError(
                    f"provider {self.id!r}: HTTP {response.status_code}: {_snippet(response)}"
                )
            return self.parse(response.json(), latency_ms)

        raise ProviderError(
            f"provider {self.id!r}: gave up after {self.spec.max_retries + 1} attempts "
            f"({last_error})"
        )

    async def _backoff(self, attempt: int, retry_after: str | None) -> None:
        if attempt >= self.spec.max_retries:
            return
        delay = min(30.0, 0.5 * 2**attempt) + random.uniform(0, 0.25)
        if retry_after:
            with contextlib.suppress(ValueError):
                delay = max(delay, float(retry_after))
        await self._sleep(delay)

    def _unexpected(self, exc: Exception) -> ProviderError:
        return ProviderError(f"provider {self.id!r}: unexpected response shape ({exc!r})")


class OpenAIProvider(HTTPProvider):
    spec: OpenAIProviderSpec

    def __init__(
        self,
        spec: OpenAIProviderSpec,
        *,
        cache: ResponseCache | None = None,
        client: httpx.AsyncClient | None = None,
        sleep: Any = asyncio.sleep,
    ) -> None:
        api_key = os.environ.get(spec.api_key_env, "")
        if not api_key and not _is_local(spec.base_url):
            raise ProviderError(
                f"provider {spec.id!r}: environment variable {spec.api_key_env} is not set"
            )
        super().__init__(spec, cache=cache, client=client, sleep=sleep)
        self._headers = {"Content-Type": "application/json"}
        if api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"

    @property
    def url(self) -> str:
        return self.spec.base_url.rstrip("/") + "/chat/completions"

    @property
    def headers(self) -> dict[str, str]:
        return self._headers

    def payload(self, request: Request) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.spec.model,
            "messages": request.messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }
        if request.tools:
            payload["tools"] = request.tools
        return payload

    def parse(self, body: dict[str, Any], latency_ms: float) -> Completion:
        try:
            message = body["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise self._unexpected(exc) from exc

        tool_calls = []
        for call in message.get("tool_calls") or []:
            fn = call.get("function", {})
            raw_args = fn.get("arguments") or "{}"
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
            except json.JSONDecodeError:
                args = {"_unparsed": raw_args}
            tool_calls.append(ToolCall(name=fn.get("name", ""), arguments=args))

        usage = body.get("usage") or {}
        input_tokens = int(usage.get("prompt_tokens", 0))
        output_tokens = int(usage.get("completion_tokens", 0))
        return Completion(
            content=message.get("content"),
            tool_calls=tool_calls,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            cost_usd=cost_of(self.spec.pricing, input_tokens, output_tokens),
        )


class MockProvider:
    def __init__(self, spec: MockProviderSpec) -> None:
        self.id = spec.id
        self.spec = spec

    async def aclose(self) -> None:
        return None

    async def complete(self, request: Request) -> Completion:
        canned = self.spec.responses.get(request.case_id, self.spec.default)
        if canned is None:
            raise ProviderError(
                f"mock provider {self.id!r} has no response for {request.case_id!r}"
            )
        if self.spec.latency_ms:
            await asyncio.sleep(self.spec.latency_ms / 1000)
        return Completion(
            content=canned.content,
            tool_calls=list(canned.tool_calls),
            input_tokens=canned.input_tokens,
            output_tokens=canned.output_tokens,
            latency_ms=self.spec.latency_ms,
            cost_usd=cost_of(self.spec.pricing, canned.input_tokens, canned.output_tokens),
        )


def build_provider(
    spec: OpenAIProviderSpec | AnthropicProviderSpec | MockProviderSpec,
    *,
    cache: ResponseCache | None = None,
) -> Provider:
    if isinstance(spec, OpenAIProviderSpec):
        return OpenAIProvider(spec, cache=cache)
    if isinstance(spec, AnthropicProviderSpec):
        from driftgate.anthropic_provider import AnthropicProvider  # optional dependency

        return AnthropicProvider(spec, cache=cache)
    return MockProvider(spec)


def _is_local(base_url: str) -> bool:
    host = httpx.URL(base_url).host
    return host in {"localhost", "127.0.0.1", "0.0.0.0", "::1"}


def _snippet(response: httpx.Response, limit: int = 300) -> str:
    text = response.text.strip().replace("\n", " ")
    return text[:limit] + ("…" if len(text) > limit else "")
