from __future__ import annotations

from pathlib import Path

import pytest

from driftgate.compare import Status, compare
from driftgate.config import Thresholds
from driftgate.report import COMMENT_MARKER, console, markdown
from driftgate.results import RunResult
from tests.conftest import case_result, run_result


def statuses(current: RunResult, baseline: RunResult | None) -> dict[str, Status]:
    return {d.case_id: d.status for d in compare(current, baseline).diffs}


def test_classifies_every_transition() -> None:
    baseline = run_result(
        case_result("stays-green", True),
        case_result("breaks", True),
        case_result("gets-fixed", False),
        case_result("stays-red", False),
        case_result("deleted", True),
    )
    current = run_result(
        case_result("stays-green", True),
        case_result("breaks", False),
        case_result("gets-fixed", True),
        case_result("stays-red", False),
        case_result("added-green", True),
        case_result("added-red", False),
    )
    assert statuses(current, baseline) == {
        "stays-green": Status.PASSING,
        "breaks": Status.REGRESSED,
        "gets-fixed": Status.FIXED,
        "stays-red": Status.STILL_FAILING,
        "deleted": Status.REMOVED,
        "added-green": Status.NEW_PASSING,
        "added-red": Status.NEW_FAILING,
    }


def test_regression_fails_gate_by_default() -> None:
    comparison = compare(run_result(case_result("a", False)), run_result(case_result("a", True)))
    assert not comparison.passed
    assert "1 regression(s)" in comparison.gate_failures[0]


def test_regressions_allowed_by_threshold() -> None:
    comparison = compare(
        run_result(case_result("a", False)),
        run_result(case_result("a", True)),
        Thresholds(max_regressions=1),
    )
    assert comparison.passed


def test_without_baseline_only_pass_rate_gates() -> None:
    current = run_result(case_result("a", True), case_result("b", False))
    assert compare(current, None).passed
    gated = compare(current, None, Thresholds(min_pass_rate=0.9))
    assert not gated.passed and "pass rate 50% below minimum 90%" in gated.gate_failures[0]


def test_cost_and_latency_budgets() -> None:
    baseline = run_result(case_result("a", True, latency_ms=100, cost_usd=0.010))
    current = run_result(case_result("a", True, latency_ms=150, cost_usd=0.013))
    comparison = compare(
        current,
        baseline,
        Thresholds(max_cost_increase_pct=20, max_p95_latency_increase_pct=40),
    )
    assert comparison.providers[0].cost_change_pct == pytest.approx(30)
    assert comparison.providers[0].p95_change_pct == pytest.approx(50)
    assert len(comparison.gate_failures) == 2


def test_markdown_report() -> None:
    baseline = run_result(case_result("a", True), case_result("b", False), sha="deadbeefcafe")
    current = run_result(case_result("a", False), case_result("b", True))
    md = markdown(compare(current, baseline))
    assert md.startswith(COMMENT_MARKER)
    assert "❌ failed" in md
    assert "### Regressions (1)" in md and "`a`" in md and "contains: nope" in md
    assert "**Fixed (1):** `p/b`" in md
    assert "baseline `deadbee`" in md
    assert "(+0 pp)" not in md


def test_console_report() -> None:
    text = console(compare(run_result(case_result("a", False)), run_result(case_result("a", True))))
    assert "REGRESSED  p/a" in text and "GATE FAILED" in text


def test_results_round_trip(tmp_path: Path) -> None:
    original = run_result(case_result("a", True), case_result("b", False))
    path = tmp_path / "nested" / "r.json"
    original.save(path)
    loaded = RunResult.load(path)
    assert loaded == original
    summary = loaded.summary()
    assert (summary.total, summary.passed, summary.pass_rate) == (2, 1, 0.5)
    assert summary.p95_latency_ms == 100


def test_rejects_unknown_schema_version(tmp_path: Path) -> None:
    path = tmp_path / "r.json"
    path.write_text(run_result().model_copy(update={"schema_version": 99}).model_dump_json())
    with pytest.raises(ValueError, match="schema v99"):
        RunResult.load(path)
