from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
import respx

from driftgate.cache import ResponseCache
from driftgate.config import MockProviderSpec, OpenAIProviderSpec
from driftgate.providers import MockProvider, OpenAIProvider, ProviderError, Request, cost_of

URL = "https://api.example.com/v1/chat/completions"
REQUEST = Request(case_id="c", messages=[{"role": "user", "content": "hi"}])


def spec(**overrides: object) -> OpenAIProviderSpec:
    base: dict[str, object] = {
        "type": "openai",
        "id": "p",
        "model": "m",
        "base_url": "https://api.example.com/v1",
        "api_key_env": "TEST_KEY",
        "pricing": {"input_per_mtok": 1.0, "output_per_mtok": 2.0},
        "max_retries": 2,
    }
    return OpenAIProviderSpec.model_validate({**base, **overrides})


def ok_body(**message: object) -> dict[str, object]:
    return {
        "choices": [{"message": {"content": "hello", **message}}],
        "usage": {"prompt_tokens": 1000, "completion_tokens": 500},
    }


async def no_sleep(_: float) -> None:
    return None


@pytest.fixture
def key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_KEY", "sk-test")


def test_missing_api_key_is_clear() -> None:
    with pytest.raises(ProviderError, match="TEST_KEY is not set"):
        OpenAIProvider(spec())


def test_local_endpoints_need_no_key() -> None:
    OpenAIProvider(spec(base_url="http://localhost:11434/v1"))


@respx.mock
async def test_parses_content_tool_calls_usage_and_cost(key: None) -> None:
    tool_calls = [
        {"function": {"name": "lookup", "arguments": json.dumps({"phone": "555"})}},
        {"function": {"name": "broken", "arguments": "{not json"}},
    ]
    route = respx.post(URL).mock(
        return_value=httpx.Response(200, json=ok_body(tool_calls=tool_calls))
    )
    provider = OpenAIProvider(spec(), sleep=no_sleep)
    result = await provider.complete(REQUEST)
    await provider.aclose()

    sent = json.loads(route.calls[0].request.content)
    assert route.calls[0].request.headers["authorization"] == "Bearer sk-test"
    assert sent["model"] == "m" and sent["temperature"] == 0.0 and "tools" not in sent
    assert result.content == "hello"
    assert [c.name for c in result.tool_calls] == ["lookup", "broken"]
    assert result.tool_calls[0].arguments == {"phone": "555"}
    assert result.tool_calls[1].arguments == {"_unparsed": "{not json"}
    assert result.cost_usd == pytest.approx((1000 * 1.0 + 500 * 2.0) / 1e6)


@respx.mock
async def test_retries_retryable_status_then_succeeds(key: None) -> None:
    slept: list[float] = []

    async def record(delay: float) -> None:
        slept.append(delay)

    route = respx.post(URL).mock(
        side_effect=[
            httpx.Response(429, headers={"retry-after": "3"}, text="slow down"),
            httpx.Response(503, text="busy"),
            httpx.Response(200, json=ok_body()),
        ]
    )
    provider = OpenAIProvider(spec(), sleep=record)
    result = await provider.complete(REQUEST)
    assert result.content == "hello"
    assert route.call_count == 3
    assert slept[0] >= 3.0  # honours Retry-After


@respx.mock
async def test_gives_up_after_max_retries(key: None) -> None:
    respx.post(URL).mock(return_value=httpx.Response(500, text="boom"))
    provider = OpenAIProvider(spec(), sleep=no_sleep)
    with pytest.raises(ProviderError, match="gave up after 3 attempts"):
        await provider.complete(REQUEST)


@respx.mock
async def test_non_retryable_error_fails_fast_without_leaking_key(key: None) -> None:
    route = respx.post(URL).mock(return_value=httpx.Response(401, text="bad key"))
    provider = OpenAIProvider(spec(), sleep=no_sleep)
    with pytest.raises(ProviderError, match="HTTP 401") as info:
        await provider.complete(REQUEST)
    assert route.call_count == 1
    assert "sk-test" not in str(info.value)


@respx.mock
async def test_retries_transport_errors(key: None) -> None:
    respx.post(URL).mock(
        side_effect=[httpx.ConnectError("reset"), httpx.Response(200, json=ok_body())]
    )
    provider = OpenAIProvider(spec(), sleep=no_sleep)
    assert (await provider.complete(REQUEST)).content == "hello"


@respx.mock
async def test_cache_hit_skips_network(key: None, tmp_path: Path) -> None:
    route = respx.post(URL).mock(return_value=httpx.Response(200, json=ok_body()))
    cache = ResponseCache(tmp_path / "c.sqlite")
    provider = OpenAIProvider(spec(), cache=cache, sleep=no_sleep)
    first = await provider.complete(REQUEST)
    second = await provider.complete(REQUEST)
    third = await provider.complete(Request(case_id="c", messages=REQUEST.messages, attempt=1))
    assert route.call_count == 2  # attempt index is part of the key
    assert not first.cached and second.cached and not third.cached
    assert second.content == first.content


@respx.mock
async def test_unexpected_shape(key: None) -> None:
    respx.post(URL).mock(return_value=httpx.Response(200, json={"oops": True}))
    with pytest.raises(ProviderError, match="unexpected response shape"):
        await OpenAIProvider(spec(), sleep=no_sleep).complete(REQUEST)


async def test_mock_provider() -> None:
    provider = MockProvider(
        MockProviderSpec.model_validate(
            {
                "type": "mock",
                "id": "m",
                "responses": {"c": {"content": "canned", "input_tokens": 10, "output_tokens": 5}},
                "pricing": {"input_per_mtok": 1, "output_per_mtok": 1},
            }
        )
    )
    result = await provider.complete(REQUEST)
    assert result.content == "canned"
    assert result.cost_usd == pytest.approx(15 / 1e6)
    with pytest.raises(ProviderError, match="no response"):
        await provider.complete(Request(case_id="other", messages=[]))


def test_cost_of_without_pricing() -> None:
    assert cost_of(None, 10, 10) is None
