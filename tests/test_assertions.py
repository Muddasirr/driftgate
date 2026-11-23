from __future__ import annotations

from typing import Any

import pytest
from pydantic import TypeAdapter

from driftgate.assertions import JudgeContext, evaluate
from driftgate.config import Assertion
from driftgate.providers import ProviderError, Request
from driftgate.results import Completion
from tests.conftest import completion

ADAPTER: TypeAdapter[Any] = TypeAdapter(Assertion)


def a(**spec: Any) -> Any:
    return ADAPTER.validate_python(spec)


@pytest.mark.parametrize(
    ("spec", "content", "passed"),
    [
        ({"type": "contains", "value": "Hello"}, "well hello there", True),
        ({"type": "contains", "value": "Hello", "case_sensitive": True}, "hello", False),
        ({"type": "not_contains", "value": "refund"}, "We will REFUND you", False),
        ({"type": "not_contains", "value": "refund"}, "We can't help", True),
        ({"type": "equals", "value": "42"}, "  42\n", True),
        ({"type": "equals", "value": "42", "strip": False}, " 42", False),
        ({"type": "regex", "pattern": r"^\d{3}$"}, "123", True),
        ({"type": "regex", "pattern": r"^\d{3}$"}, "12a", False),
        ({"type": "json_valid"}, '{"a": 1}', True),
        ({"type": "json_valid"}, '```json\n{"a": 1}\n```', True),
        ({"type": "json_valid"}, "{a: 1}", False),
    ],
)
async def test_text_assertions(spec: dict[str, Any], content: str, passed: bool) -> None:
    outcome = await evaluate(a(**spec), completion(content))
    assert outcome.passed is passed
    assert (outcome.detail == "") is passed


async def test_json_schema_reports_path() -> None:
    schema = {
        "type": "object",
        "properties": {"age": {"type": "integer"}},
        "required": ["age"],
    }
    ok = await evaluate(a(type="json_schema", schema=schema), completion('{"age": 3}'))
    bad = await evaluate(a(type="json_schema", schema=schema), completion('{"age": "3"}'))
    invalid = await evaluate(a(type="json_schema", schema=schema), completion("nope"))
    assert ok.passed
    assert not bad.passed and bad.detail.startswith("age:")
    assert not invalid.passed and "not valid JSON" in invalid.detail


async def test_tool_call_matches_name_and_args() -> None:
    out = completion(None, [("lookup_patient", {"phone": "5551234567", "region": "US"})])
    exact = a(type="tool_call", name="lookup_patient", args={"region": "US"})
    regex = a(type="tool_call", name="lookup_patient", args={"phone": {"regex": r"^\d{10}$"}})
    wrong = a(type="tool_call", name="lookup_patient", args={"region": "PK"})
    missing_arg = a(type="tool_call", name="lookup_patient", args={"email": "x"})
    other = a(type="tool_call", name="create_patient")
    assert (await evaluate(exact, out)).passed
    assert (await evaluate(regex, out)).passed
    assert "expected 'PK'" in (await evaluate(wrong, out)).detail
    assert "missing 'email'" in (await evaluate(missing_arg, out)).detail
    assert "expected call to create_patient" in (await evaluate(other, out)).detail


async def test_tool_call_position_first() -> None:
    out = completion(None, [("search", {}), ("lookup", {})])
    assert not (await evaluate(a(type="tool_call", name="lookup", position="first"), out)).passed
    assert (await evaluate(a(type="tool_call", name="search", position="first"), out)).passed


async def test_no_tool_call() -> None:
    calls = completion(None, [("create_patient", {})])
    assert not (await evaluate(a(type="no_tool_call"), calls)).passed
    assert not (await evaluate(a(type="no_tool_call", name="create_patient"), calls)).passed
    assert (await evaluate(a(type="no_tool_call", name="delete_patient"), calls)).passed
    assert (await evaluate(a(type="no_tool_call"), completion("just text"))).passed


async def test_latency_and_cost_budgets() -> None:
    fast = completion("x", latency_ms=200, cost_usd=0.001)
    slow = completion("x", latency_ms=900, cost_usd=0.02)
    assert (await evaluate(a(type="max_latency_ms", value=500), fast)).passed
    assert not (await evaluate(a(type="max_latency_ms", value=500), slow)).passed
    assert (await evaluate(a(type="max_cost_usd", value=0.01), fast)).passed
    assert not (await evaluate(a(type="max_cost_usd", value=0.01), slow)).passed


async def test_latency_skipped_for_cached_and_cost_needs_pricing() -> None:
    cached = completion("x", latency_ms=99999, cached=True)
    assert (await evaluate(a(type="max_latency_ms", value=1), cached)).passed
    unpriced = await evaluate(a(type="max_cost_usd", value=1), completion("x", cost_usd=None))
    assert not unpriced.passed and "no pricing" in unpriced.detail


class FakeJudge:
    id = "judge"

    def __init__(self, reply: str | Exception) -> None:
        self.reply = reply
        self.requests: list[Request] = []

    async def complete(self, request: Request) -> Completion:
        self.requests.append(request)
        if isinstance(self.reply, Exception):
            raise self.reply
        return Completion(content=self.reply)

    async def aclose(self) -> None:
        return None


def ctx(judge: FakeJudge) -> JudgeContext:
    return JudgeContext(
        providers={"judge": judge}, default_judge="judge", case_id="c", conversation="USER: hi"
    )


@pytest.mark.parametrize(
    ("reply", "min_score", "passed", "detail"),
    [
        ('{"score": 5, "reason": "great"}', 4, True, "score 5/5"),
        ('Sure! {"score": 3, "reason": "meh"}', 4, False, "score 3/5"),
        ('{"score": 9}', 4, False, "unparseable"),
        ("I think it's fine", 4, False, "unparseable"),
    ],
)
async def test_llm_judge(reply: str, min_score: int, passed: bool, detail: str) -> None:
    judge = FakeJudge(reply)
    outcome = await evaluate(
        a(type="llm_judge", rubric="polite", min_score=min_score),
        completion("hello", [("lookup", {"x": 1})]),
        ctx(judge),
    )
    assert outcome.passed is passed
    assert detail in outcome.detail
    prompt = judge.requests[0].messages[1]["content"]
    assert "RUBRIC:\npolite" in prompt and "lookup" in prompt and "USER: hi" in prompt


async def test_llm_judge_failures_are_reported() -> None:
    err = await evaluate(
        a(type="llm_judge", rubric="r"), completion("x"), ctx(FakeJudge(ProviderError("down")))
    )
    assert not err.passed and "judge failed: down" in err.detail
    no_ctx = await evaluate(a(type="llm_judge", rubric="r"), completion("x"), None)
    assert not no_ctx.passed
