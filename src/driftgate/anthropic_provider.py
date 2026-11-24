"""Claude models through the official Anthropic SDK.

Install with `pip install "driftgate[anthropic]"`. The SDK owns authentication,
the API version header and retries (including 429 and 529 overloaded); this module
translates driftgate's OpenAI-style cases into Messages API requests and maps the
response back.

Two deliberate choices for evaluation:
- No sampling parameters are sent. Current Claude models reject `temperature`.
- No refusal fallbacks. A fallback would answer with a different model and the run
  would measure the wrong thing, so a refusal is recorded as a failed attempt.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

from driftgate.cache import ResponseCache
from driftgate.config import AnthropicProviderSpec, ToolCall
from driftgate.providers import ProviderError, Request, cost_of
from driftgate.results import Completion

try:
    import anthropic
except ImportError:  # pragma: no cover - exercised only without the extra installed
    anthropic = None  # type: ignore[assignment]


class AnthropicProvider:
    def __init__(
        self,
        spec: AnthropicProviderSpec,
        *,
        cache: ResponseCache | None = None,
        client: Any = None,
    ) -> None:
        self.id = spec.id
        self.spec = spec
        self._cache = cache
        if client is not None:
            self._client = client
            return
        if anthropic is None:
            raise ProviderError(
                f"provider {spec.id!r}: the Anthropic SDK is not installed. "
                'Run: pip install "driftgate[anthropic]"'
            )
        try:
            # With no key in the environment, the SDK falls back to its other
            # credential sources (auth token, `ant auth login` profile).
            self._client = anthropic.AsyncAnthropic(
                api_key=os.environ.get(spec.api_key_env) or None,
                base_url=spec.base_url,
                max_retries=spec.max_retries,
                timeout=spec.timeout_s,
            )
        except anthropic.AnthropicError as exc:
            raise ProviderError(f"provider {spec.id!r}: {exc}") from exc

    async def aclose(self) -> None:
        await self._client.close()

    def payload(self, request: Request) -> dict[str, Any]:
        system, messages = to_anthropic_messages(request.messages)
        payload: dict[str, Any] = {
            "model": self.spec.model,
            "max_tokens": self.spec.max_tokens or request.max_tokens,
            "messages": messages,
        }
        if system:
            payload["system"] = system
        if request.tools:
            payload["tools"] = to_anthropic_tools(request.tools)
        if self.spec.effort:
            payload["output_config"] = {"effort": self.spec.effort}
        return payload

    async def complete(self, request: Request) -> Completion:
        payload = self.payload(request)
        cache_key = None
        if self._cache is not None:
            cache_key = self._cache.key(
                {"provider": "anthropic", "attempt": request.attempt, **payload}
            )
            hit = self._cache.get(cache_key)
            if hit is not None:
                return hit.model_copy(update={"cached": True})

        started = time.perf_counter()
        try:
            response = await self._client.messages.create(**payload)
        except Exception as exc:
            raise self._error(exc) from exc
        completion = self.parse(response, (time.perf_counter() - started) * 1000)

        if self._cache is not None and cache_key is not None:
            self._cache.put(cache_key, completion)
        return completion

    def parse(self, response: Any, latency_ms: float) -> Completion:
        if response.stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            category = getattr(details, "category", None) or "unspecified"
            raise ProviderError(f"provider {self.id!r}: model refused (category: {category})")

        texts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in response.content:
            if block.type == "text":
                texts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(name=block.name, arguments=dict(block.input or {})))
            # thinking blocks are the model's reasoning, not its answer: skipped

        if response.stop_reason == "max_tokens" and not texts and not tool_calls:
            raise ProviderError(
                f"provider {self.id!r}: hit max_tokens before answering; "
                "raise max_tokens on this provider (thinking counts toward it)"
            )

        input_tokens = int(response.usage.input_tokens or 0)
        output_tokens = int(response.usage.output_tokens or 0)
        return Completion(
            content="".join(texts) or None,
            tool_calls=tool_calls,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            cost_usd=cost_of(self.spec.pricing, input_tokens, output_tokens),
        )

    def _error(self, exc: Exception) -> ProviderError:
        prefix = f"provider {self.id!r}"
        if anthropic is not None:
            if isinstance(exc, anthropic.APIStatusError):
                return ProviderError(f"{prefix}: HTTP {exc.status_code}: {exc.message}")
            if isinstance(exc, anthropic.APIConnectionError):
                return ProviderError(f"{prefix}: connection failed: {exc}")
            if isinstance(exc, anthropic.AnthropicError):
                return ProviderError(f"{prefix}: {exc}")
        return ProviderError(f"{prefix}: {type(exc).__name__}: {exc}")


def to_anthropic_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """OpenAI `{"type": "function", "function": {...}}` tools to Anthropic tools."""
    converted = []
    for tool in tools:
        fn = tool.get("function", tool)
        entry: dict[str, Any] = {
            "name": fn["name"],
            "input_schema": fn.get("parameters") or {"type": "object", "properties": {}},
        }
        if fn.get("description"):
            entry["description"] = fn["description"]
        converted.append(entry)
    return converted


def to_anthropic_messages(
    messages: list[dict[str, Any]],
) -> tuple[str | None, list[dict[str, Any]]]:
    """Convert OpenAI-format messages to (system, messages) for the Messages API.

    System turns move to the top-level `system` field, assistant tool calls become
    `tool_use` blocks, and tool results become `tool_result` blocks. Consecutive turns
    with the same role are merged, so the results of parallel tool calls arrive in one
    user message, as the API expects.
    """
    system_parts: list[str] = []
    converted: list[dict[str, Any]] = []

    def append(role: str, blocks: list[dict[str, Any]]) -> None:
        if converted and converted[-1]["role"] == role:
            converted[-1]["content"].extend(blocks)
        else:
            converted.append({"role": role, "content": blocks})

    for message in messages:
        role = message["role"]
        if role == "system":
            if message.get("content"):
                system_parts.append(message["content"])
        elif role == "tool":
            append(
                "user",
                [
                    {
                        "type": "tool_result",
                        "tool_use_id": message["tool_call_id"],
                        "content": message.get("content") or "",
                    }
                ],
            )
        else:
            blocks: list[dict[str, Any]] = []
            if message.get("content"):
                blocks.append({"type": "text", "text": message["content"]})
            for call in message.get("tool_calls") or []:
                fn = call["function"]
                try:
                    arguments = json.loads(fn.get("arguments") or "{}")
                except json.JSONDecodeError:
                    arguments = {}
                blocks.append(
                    {"type": "tool_use", "id": call["id"], "name": fn["name"], "input": arguments}
                )
            append(role, blocks)

    return ("\n\n".join(system_parts) or None), converted
