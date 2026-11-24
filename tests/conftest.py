from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from driftgate.config import ToolCall
from driftgate.results import AssertionOutcome, Attempt, CaseResult, Completion, RunResult


def write_suite(tmp_path: Path, data: dict[str, Any], name: str = "suite.yaml") -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def mock_suite(
    responses: dict[str, Any], cases: list[dict[str, Any]], **extra: Any
) -> dict[str, Any]:
    return {
        "name": "test-suite",
        "providers": [{"id": "mock", "type": "mock", "responses": responses}],
        "cases": cases,
        **extra,
    }


def completion(
    content: str | None = "",
    tool_calls: list[tuple[str, dict[str, Any]]] | None = None,
    latency_ms: float = 100.0,
    cost_usd: float | None = None,
    cached: bool = False,
) -> Completion:
    return Completion(
        content=content,
        tool_calls=[ToolCall(name=n, arguments=a) for n, a in (tool_calls or [])],
        latency_ms=latency_ms,
        cost_usd=cost_usd,
        cached=cached,
    )


def case_result(
    case_id: str,
    passed: bool,
    provider_id: str = "p",
    latency_ms: float = 100.0,
    cost_usd: float | None = 0.001,
) -> CaseResult:
    outcome = AssertionOutcome(type="contains", passed=passed, detail="" if passed else "nope")
    return CaseResult(
        case_id=case_id,
        provider_id=provider_id,
        attempts=[
            Attempt(
                completion=completion("x", latency_ms=latency_ms, cost_usd=cost_usd),
                assertions=[outcome],
            )
        ],
    )


def run_result(*results: CaseResult, sha: str = "abc1234") -> RunResult:
    return RunResult(
        suite="s",
        driftgate_version="test",
        started_at="2026-01-01T00:00:00+00:00",
        finished_at="2026-01-01T00:00:01+00:00",
        git_sha=sha,
        results=list(results),
    )


@pytest.fixture(autouse=True)
def _no_real_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GROQ_API_KEY", "GITHUB_STEP_SUMMARY"):
        monkeypatch.delenv(var, raising=False)
