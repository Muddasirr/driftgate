"""Assertion evaluation.

Deterministic checks run first and cheaply; `llm_judge` calls a judge model and
is the only assertion that can cost money.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import jsonschema

from driftgate.config import (
    Assertion,
    ContainsAssertion,
    EqualsAssertion,
    JsonSchemaAssertion,
    JsonValidAssertion,
    LLMJudgeAssertion,
    MaxCostAssertion,
    MaxLatencyAssertion,
    NotContainsAssertion,
    NoToolCallAssertion,
    RegexAssertion,
    ToolCallAssertion,
)
from driftgate.providers import Provider, ProviderError, Request
from driftgate.results import AssertionOutcome, Completion

JUDGE_SYSTEM_PROMPT = (
    "You are a strict evaluator of AI assistant outputs. Score the OUTPUT against the "
    "RUBRIC from 1 (fails completely) to 5 (fully meets it). Judge only what the rubric "
    'asks. Reply with JSON only: {"score": <1-5>, "reason": "<one sentence>"}'
)
_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


@dataclass(frozen=True)
class JudgeContext:
    providers: Mapping[str, Provider]
    default_judge: str | None
    case_id: str
    conversation: str


async def evaluate(
    assertion: Assertion, completion: Completion, judge: JudgeContext | None = None
) -> AssertionOutcome:
    text = completion.content or ""
    match assertion:
        case ContainsAssertion():
            ok = _norm(assertion.value, assertion.case_sensitive) in _norm(
                text, assertion.case_sensitive
            )
            return _outcome(assertion.type, ok, f"expected to contain {assertion.value!r}")
        case NotContainsAssertion():
            ok = _norm(assertion.value, assertion.case_sensitive) not in _norm(
                text, assertion.case_sensitive
            )
            return _outcome(assertion.type, ok, f"must not contain {assertion.value!r}")
        case EqualsAssertion():
            got = text.strip() if assertion.strip else text
            want = assertion.value.strip() if assertion.strip else assertion.value
            return _outcome(assertion.type, got == want, f"expected {want!r}, got {_clip(got)!r}")
        case RegexAssertion():
            ok = re.search(assertion.pattern, text) is not None
            return _outcome(assertion.type, ok, f"no match for /{assertion.pattern}/")
        case JsonValidAssertion():
            parsed = _parse_json(text)
            return _outcome(assertion.type, parsed is not _INVALID, "output is not valid JSON")
        case JsonSchemaAssertion():
            return _check_schema(assertion, text)
        case ToolCallAssertion():
            return _check_tool_call(assertion, completion)
        case NoToolCallAssertion():
            names = [c.name for c in completion.tool_calls]
            ok = not names if assertion.name is None else assertion.name not in names
            target = assertion.name or "any tool"
            return _outcome(assertion.type, ok, f"called {target} (calls: {names})")
        case MaxLatencyAssertion():
            if completion.cached:
                return AssertionOutcome(type=assertion.type, passed=True, detail="skipped: cached")
            ok = completion.latency_ms <= assertion.value
            detail = f"{completion.latency_ms:.0f} ms > {assertion.value:.0f} ms"
            return _outcome(assertion.type, ok, detail)
        case MaxCostAssertion():
            if completion.cost_usd is None:
                return AssertionOutcome(
                    type=assertion.type, passed=False, detail="no pricing configured for provider"
                )
            ok = completion.cost_usd <= assertion.value
            detail = f"${completion.cost_usd:.6f} > ${assertion.value:.6f}"
            return _outcome(assertion.type, ok, detail)
        case LLMJudgeAssertion():
            return await _judge(assertion, completion, judge)
    raise AssertionError(f"unhandled assertion type: {assertion!r}")  # pragma: no cover


def _outcome(kind: str, passed: bool, failure_detail: str) -> AssertionOutcome:
    return AssertionOutcome(type=kind, passed=passed, detail="" if passed else failure_detail)


def _norm(value: str, case_sensitive: bool) -> str:
    return value if case_sensitive else value.casefold()


def _clip(value: str, limit: int = 120) -> str:
    return value if len(value) <= limit else value[:limit] + "…"


_INVALID = object()


def _parse_json(text: str) -> Any:
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```[a-zA-Z]*\n?|```$", "", candidate).strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return _INVALID


def _check_schema(assertion: JsonSchemaAssertion, text: str) -> AssertionOutcome:
    parsed = _parse_json(text)
    if parsed is _INVALID:
        return _outcome(assertion.type, False, "output is not valid JSON")
    try:
        jsonschema.validate(parsed, assertion.json_schema)
    except jsonschema.ValidationError as exc:
        path = "/".join(str(p) for p in exc.absolute_path) or "<root>"
        return _outcome(assertion.type, False, f"{path}: {exc.message}")
    return _outcome(assertion.type, True, "")


def _check_tool_call(assertion: ToolCallAssertion, completion: Completion) -> AssertionOutcome:
    calls = completion.tool_calls
    names = [c.name for c in calls]
    candidates = calls[:1] if assertion.position == "first" else calls
    matching = [c for c in candidates if c.name == assertion.name]
    if not matching:
        where = "as the first call" if assertion.position == "first" else ""
        return _outcome(
            assertion.type, False, f"expected call to {assertion.name} {where}(calls: {names})"
        )
    problems: list[str] = []
    for call in matching:
        problems = _arg_mismatches(assertion.args, call.arguments)
        if not problems:
            return _outcome(assertion.type, True, "")
    return _outcome(assertion.type, False, f"{assertion.name} args: " + "; ".join(problems))


def _arg_mismatches(expected: dict[str, Any], actual: dict[str, Any]) -> list[str]:
    problems = []
    for key, want in expected.items():
        if key not in actual:
            problems.append(f"missing {key!r}")
            continue
        got = actual[key]
        if isinstance(want, dict) and set(want) == {"regex"}:
            if re.search(str(want["regex"]), str(got)) is None:
                problems.append(f"{key}={got!r} does not match /{want['regex']}/")
        elif got != want:
            problems.append(f"{key}={got!r}, expected {want!r}")
    return problems


async def _judge(
    assertion: LLMJudgeAssertion, completion: Completion, ctx: JudgeContext | None
) -> AssertionOutcome:
    if ctx is None:
        return AssertionOutcome(type=assertion.type, passed=False, detail="no judge configured")
    judge_id = assertion.provider or ctx.default_judge
    provider = ctx.providers.get(judge_id or "")
    if provider is None:
        return AssertionOutcome(
            type=assertion.type, passed=False, detail=f"judge {judge_id!r} unavailable"
        )

    output = completion.content or ""
    if completion.tool_calls:
        calls = ", ".join(f"{c.name}({json.dumps(c.arguments)})" for c in completion.tool_calls)
        output = f"{output}\n[tool calls: {calls}]".strip()
    user = f"RUBRIC:\n{assertion.rubric}\n\nCONVERSATION:\n{ctx.conversation}\n\nOUTPUT:\n{output}"
    request = Request(
        case_id=f"judge:{ctx.case_id}",
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        temperature=0.0,
        max_tokens=200,
    )
    try:
        verdict = await provider.complete(request)
    except ProviderError as exc:
        return AssertionOutcome(type=assertion.type, passed=False, detail=f"judge failed: {exc}")

    score, reason = _parse_verdict(verdict.content or "")
    if score is None:
        return AssertionOutcome(
            type=assertion.type, passed=False, detail=f"unparseable verdict: {_clip(reason)}"
        )
    passed = score >= assertion.min_score
    return AssertionOutcome(
        type=assertion.type,
        passed=passed,
        detail=f"score {score}/5 (min {assertion.min_score}): {reason}",
    )


def _parse_verdict(text: str) -> tuple[int | None, str]:
    found = _JSON_OBJECT.search(text)
    if not found:
        return None, text
    try:
        data = json.loads(found.group(0))
        score = int(data["score"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None, text
    if not 1 <= score <= 5:
        return None, text
    return score, str(data.get("reason", "")).strip()
