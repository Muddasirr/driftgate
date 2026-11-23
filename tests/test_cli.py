"""End-to-end through the CLI with the mock provider: no network, no keys."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from driftgate.cli import main
from tests.conftest import mock_suite, write_suite

CASES = [
    {"id": "greets", "input": "hi", "assert": [{"type": "contains", "value": "hello"}]},
    {
        "id": "looks-up",
        "input": "my number is 555 123 4567",
        "assert": [{"type": "tool_call", "name": "lookup", "args": {"phone": "5551234567"}}],
        "repeat": 3,
    },
]
GOOD = {
    "greets": {"content": "hello!"},
    "looks-up": {"tool_calls": [{"name": "lookup", "arguments": {"phone": "5551234567"}}]},
}


def invoke(*args: str) -> Any:
    return CliRunner().invoke(main, list(args), catch_exceptions=False)


def test_run_then_regression_against_baseline(tmp_path: Path) -> None:
    good = write_suite(tmp_path, mock_suite(GOOD, CASES), "good.yaml")
    base = tmp_path / "base.json"
    first = invoke("run", str(good), "--out", str(base), "--no-cache")
    assert first.exit_code == 0, first.output
    assert "2/2 passed" in first.output and "Gate passed." in first.output

    saved = json.loads(base.read_text())
    assert saved["schema_version"] == 1 and len(saved["results"]) == 2
    looks_up = next(r for r in saved["results"] if r["case_id"] == "looks-up")
    assert len(looks_up["attempts"]) == 3

    broken = {**GOOD, "greets": {"content": "go away"}}
    bad = write_suite(tmp_path, mock_suite(broken, CASES), "bad.yaml")
    report = tmp_path / "report.md"
    second = invoke(
        "run", str(bad), "--out", str(tmp_path / "cur.json"), "--baseline", str(base),
        "--markdown", str(report), "--no-cache",
    )  # fmt: skip
    assert second.exit_code == 1
    assert "REGRESSED  mock/greets" in second.output
    assert "### Regressions (1)" in report.read_text()


def test_compare_command_uses_suite_thresholds(tmp_path: Path) -> None:
    suite = write_suite(
        tmp_path, mock_suite(GOOD, CASES, thresholds={"max_regressions": 5}), "s.yaml"
    )
    base, cur = tmp_path / "b.json", tmp_path / "c.json"
    invoke("run", str(suite), "--out", str(base), "--no-cache")
    invoke("run", str(suite), "--out", str(cur), "--no-cache")
    result = invoke("compare", str(base), str(cur), "--suite", str(suite))
    assert result.exit_code == 0 and "Gate passed." in result.output


def test_filters_by_tag_and_case(tmp_path: Path) -> None:
    cases = [
        {**CASES[0], "tags": ["smoke"]},
        CASES[1],
    ]
    suite = write_suite(tmp_path, mock_suite(GOOD, cases))
    out = tmp_path / "r.json"
    invoke("run", str(suite), "--tag", "smoke", "--out", str(out), "--no-cache")
    assert [r["case_id"] for r in json.loads(out.read_text())["results"]] == ["greets"]
    bad = invoke("run", str(suite), "--case", "nope", "--out", str(out), "--no-cache")
    assert bad.exit_code == 2 and "unknown case id" in bad.output


def test_writes_github_step_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    suite = write_suite(tmp_path, mock_suite(GOOD, CASES))
    invoke("run", str(suite), "--out", str(tmp_path / "r.json"), "--no-cache")
    assert "driftgate · test-suite · ✅ passed" in summary.read_text()


def test_missing_baseline_is_not_an_error(tmp_path: Path) -> None:
    suite = write_suite(tmp_path, mock_suite(GOOD, CASES))
    result = invoke(
        "run", str(suite), "--out", str(tmp_path / "r.json"),
        "--baseline", str(tmp_path / "none.json"), "--no-cache",
    )  # fmt: skip
    assert result.exit_code == 0
    assert "No baseline" in result.output


def test_missing_api_key_exits_2(tmp_path: Path) -> None:
    data = {
        "name": "s",
        "providers": [{"id": "real", "type": "openai", "model": "m", "api_key_env": "NOPE_KEY"}],
        "cases": CASES[:1],
    }
    result = invoke("run", str(write_suite(tmp_path, data)), "--out", str(tmp_path / "r.json"))
    assert result.exit_code == 2 and "NOPE_KEY is not set" in result.output


def test_validate_and_init(tmp_path: Path) -> None:
    path = tmp_path / "driftgate.yaml"
    assert invoke("init", str(path)).exit_code == 0
    validated = invoke("validate", str(path))
    assert validated.exit_code == 0 and "2 cases" in validated.output
    assert invoke("init", str(path)).exit_code == 2
    broken = tmp_path / "broken.yaml"
    broken.write_text("name: x\nproviders: []\ncases: []\n")
    result = invoke("validate", str(broken))
    assert result.exit_code == 2 and "invalid suite" in result.output


def test_concurrency_override(tmp_path: Path) -> None:
    suite = write_suite(tmp_path, mock_suite(GOOD, CASES))
    ok = invoke("run", str(suite), "--concurrency", "1", "--out", str(tmp_path / "r.json"))
    assert ok.exit_code == 0
    bad = invoke("run", str(suite), "--concurrency", "0", "--out", str(tmp_path / "r.json"))
    assert bad.exit_code == 2
