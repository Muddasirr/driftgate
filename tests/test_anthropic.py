from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import anthropic
import httpx2
import pytest

from driftgate.anthropic_provider import (
    AnthropicProvider,
    to_anthropic_messages,
    to_anthropic_tools,
)
from driftgate.cache import ResponseCache
from driftgate.config import AnthropicProviderSpec, load_suite
from driftgate.providers import ProviderError, Request, build_provider
from tests.conftest import write_suite

REQUEST = Request(case_id="c", messages=[{"role": "user", "content": "hi"}])


def spec(**overrides: object) -> AnthropicProviderSpec:
    base: dict[str, object] = {
        "type": "anthropic",
        "id": "claude",
        "model": "claude-opus-5-5",
        "pricing": {"input_per_mtok": 4.0, "output_per_mtok": 20.0},
    }
    return AnthropicProviderSpec.model_validate({**base, **overrides})


def block(kind: str, **fields: Any) -> SimpleNamespace:
    return SimpleNamespace(type=kind, **fields)


def message(*blocks: SimpleNamespace, stop: str = "end_turn", **extra: Any) -> SimpleNamespace:
    return SimpleNamespace(
        content=list(blocks),
        stop_reason=stop,
        usage=SimpleNamespace(input_tokens=1000, output_tokens=100),
        **extra,
    )


class FakeClient:
    """Stands in for anthropic.AsyncAnthropic: records calls, returns or raises."""

    def __init__(self, *results: Any) -> None:
        self.results = list(results)
        self.calls: list[dict[str, Any]] = []
        self.messages = SimpleNamespace(create=self._create)
        self.closed = False

    async def _create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    async def close(self) -> None:
        self.closed = True


def test_converts_history_system_tool_calls_and_results() -> None:
    messages = [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "I'm Jane, 555 123 4567"},
        {
            "role": "assistant",
            "content": "Let me check.",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": '{"phone": "5551234567"}'},
                },
                {"id": "c2", "type": "function", "function": {"name": "slots", "arguments": ""}},
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": '{"found": true}'},
        {"role": "tool", "tool_call_id": "c2", "content": "[]"},
        {"role": "user", "content": "So?"},
    ]
    system, converted = to_anthropic_messages(messages)
    assert system == "Be brief."
    assert [m["role"] for m in converted] == ["user", "assistant", "user"]
    assistant = converted[1]["content"]
    assert assistant[0] == {"type": "text", "text": "Let me check."}
    assert assistant[1] == {
        "type": "tool_use",
        "id": "c1",
        "name": "lookup",
        "input": {"phone": "5551234567"},
    }
    assert assistant[2]["input"] == {}
    # Results of parallel tool calls and the next user turn share one user message.
    assert converted[2]["content"] == [
        {"type": "tool_result", "tool_use_id": "c1", "content": '{"found": true}'},
        {"type": "tool_result", "tool_use_id": "c2", "content": "[]"},
        {"type": "text", "text": "So?"},
    ]


def test_converts_tools() -> None:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "Find a patient.",
                "parameters": {"type": "object", "properties": {"phone": {"type": "string"}}},
            },
        },
        {"type": "function", "function": {"name": "slots"}},
    ]
    assert to_anthropic_tools(tools) == [
        {
            "name": "lookup",
            "description": "Find a patient.",
            "input_schema": {"type": "object", "properties": {"phone": {"type": "string"}}},
        },
        {"name": "slots", "input_schema": {"type": "object", "properties": {}}},
    ]


async def test_request_shape_and_parsing() -> None:
    client = FakeClient(
        message(
            block("thinking", thinking=""),
            block("text", text="Checking "),
            block("text", text="now."),
            block("tool_use", id="t1", name="lookup", input={"phone": "555"}),
        )
    )
    provider = AnthropicProvider(spec(max_tokens=16000, effort="low"), client=client)
    request = Request(
        case_id="c",
        messages=[{"role": "system", "content": "S"}, {"role": "user", "content": "hi"}],
        tools=[{"type": "function", "function": {"name": "lookup", "parameters": {}}}],
        temperature=0.7,
        max_tokens=300,
    )
    result = await provider.complete(request)
    await provider.aclose()

    sent = client.calls[0]
    assert sent["model"] == "claude-opus-5-5"
    assert sent["system"] == "S"
    assert sent["max_tokens"] == 16000  # provider override beats the suite default
    assert sent["output_config"] == {"effort": "low"}
    assert sent["tools"][0]["name"] == "lookup"
    assert "temperature" not in sent  # current Claude models reject sampling params
    assert result.content == "Checking now."
    assert [(c.name, c.arguments) for c in result.tool_calls] == [("lookup", {"phone": "555"})]
    assert result.cost_usd == pytest.approx((1000 * 4 + 100 * 20) / 1e6)
    assert client.closed


async def test_defaults_omit_optional_fields() -> None:
    client = FakeClient(message(block("text", text="ok")))
    await AnthropicProvider(spec(), client=client).complete(REQUEST)
    sent = client.calls[0]
    assert sent["max_tokens"] == 512
    assert "system" not in sent and "tools" not in sent and "output_config" not in sent


async def test_refusal_is_an_error_not_a_fallback() -> None:
    refused = message(stop="refusal", stop_details=SimpleNamespace(category="cyber"))
    provider = AnthropicProvider(spec(), client=FakeClient(refused))
    with pytest.raises(ProviderError, match=r"refused \(category: cyber\)"):
        await provider.complete(REQUEST)


async def test_truncated_thinking_gives_actionable_error() -> None:
    cut = message(block("thinking", thinking=""), stop="max_tokens")
    with pytest.raises(ProviderError, match="raise max_tokens"):
        await AnthropicProvider(spec(), client=FakeClient(cut)).complete(REQUEST)


async def test_maps_sdk_errors() -> None:
    req = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    status = anthropic.APIStatusError(
        "overloaded", response=httpx2.Response(529, request=req), body=None
    )
    conn = anthropic.APIConnectionError(request=req)
    provider = AnthropicProvider(spec(), client=FakeClient(status, conn, RuntimeError("boom")))
    with pytest.raises(ProviderError, match="HTTP 529: overloaded"):
        await provider.complete(REQUEST)
    with pytest.raises(ProviderError, match="connection failed"):
        await provider.complete(REQUEST)
    with pytest.raises(ProviderError, match="RuntimeError: boom"):
        await provider.complete(REQUEST)


async def test_cache(tmp_path: Path) -> None:
    client = FakeClient(message(block("text", text="a")), message(block("text", text="b")))
    provider = AnthropicProvider(spec(), client=client, cache=ResponseCache(tmp_path / "c.sqlite"))
    first = await provider.complete(REQUEST)
    second = await provider.complete(REQUEST)
    assert len(client.calls) == 1
    assert second.cached and second.content == first.content == "a"


def test_builds_real_sdk_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    data = {
        "name": "s",
        "providers": [
            {"id": "claude", "type": "anthropic", "model": "claude-opus-5-5", "max_retries": 3}
        ],
        "cases": [{"id": "a", "input": "x", "assert": [{"type": "json_valid"}]}],
    }
    provider = build_provider(load_suite(write_suite(tmp_path, data)).providers[0])
    assert isinstance(provider, AnthropicProvider)
    assert provider._client.max_retries == 3
