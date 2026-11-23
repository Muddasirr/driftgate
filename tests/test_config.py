from __future__ import annotations

import json
from pathlib import Path

import pytest

from driftgate.config import ConfigError, LLMJudgeAssertion, ToolCallAssertion, load_suite
from tests.conftest import mock_suite, write_suite

CASE = {"id": "a", "input": "hi", "assert": [{"type": "contains", "value": "hi"}]}


def test_loads_minimal_suite(tmp_path: Path) -> None:
    suite = load_suite(write_suite(tmp_path, mock_suite({}, [CASE])))
    assert suite.name == "test-suite"
    assert suite.cases[0].id == "a"
    assert suite.defaults.temperature == 0.0


def test_resolves_file_refs_relative_to_suite(tmp_path: Path) -> None:
    (tmp_path / "prompt.md").write_text("Be brief.", encoding="utf-8")
    tools = [{"type": "function", "function": {"name": "lookup", "parameters": {}}}]
    (tmp_path / "tools.json").write_text(json.dumps(tools), encoding="utf-8")
    data = mock_suite({}, [CASE], system="file://prompt.md", tools="file://tools.json")
    suite = load_suite(write_suite(tmp_path, data))
    assert suite.system == "Be brief."
    assert suite.tools == tools


def test_missing_file_ref_is_a_config_error(tmp_path: Path) -> None:
    data = mock_suite({}, [CASE], system="file://nope.md")
    with pytest.raises(ConfigError, match="referenced file not found"):
        load_suite(write_suite(tmp_path, data))


def test_parses_all_assertion_types(tmp_path: Path) -> None:
    case = {
        "id": "everything",
        "input": "x",
        "assert": [
            {"type": "contains", "value": "a"},
            {"type": "not_contains", "value": "b"},
            {"type": "equals", "value": "c"},
            {"type": "regex", "pattern": "^d+$"},
            {"type": "json_valid"},
            {"type": "json_schema", "schema": {"type": "object"}},
            {"type": "tool_call", "name": "f", "args": {"x": {"regex": "1"}}, "position": "first"},
            {"type": "no_tool_call"},
            {"type": "max_latency_ms", "value": 500},
            {"type": "max_cost_usd", "value": 0.01},
            {"type": "llm_judge", "rubric": "good", "provider": "mock"},
        ],
    }
    suite = load_suite(write_suite(tmp_path, mock_suite({}, [case])))
    kinds = [a.type for a in suite.cases[0].assertions]
    assert kinds[-1] == "llm_judge"
    assert isinstance(suite.cases[0].assertions[6], ToolCallAssertion)
    assert isinstance(suite.cases[0].assertions[-1], LLMJudgeAssertion)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"cases": [CASE, CASE]}, "duplicate case id"),
        ({"judge": "missing"}, "judge 'missing' is not a provider id"),
        (
            {"cases": [{"id": "a", "assert": [{"type": "contains", "value": "x"}]}]},
            "exactly one of 'input' or 'messages'",
        ),
        (
            {"cases": [{"id": "Bad Id", "input": "x", "assert": [{"type": "json_valid"}]}]},
            "must be lowercase",
        ),
        (
            {"cases": [{"id": "a", "input": "x", "assert": [{"type": "regex", "pattern": "("}]}]},
            "invalid regex",
        ),
        (
            {
                "cases": [
                    {"id": "a", "input": "x", "assert": [{"type": "llm_judge", "rubric": "r"}]}
                ]
            },
            "llm_judge needs 'provider'",
        ),
        ({"cases": [{"id": "a", "input": "x", "assert": [{"type": "nope"}]}]}, "assert"),
        ({"cases": [{"id": "a", "input": "x", "assert": []}]}, "assert"),
    ],
)
def test_rejects_invalid_suites(tmp_path: Path, change: dict[str, object], message: str) -> None:
    data = {**mock_suite({}, [CASE]), **change}
    with pytest.raises(ConfigError, match=message):
        load_suite(write_suite(tmp_path, data))


def test_rejects_non_mapping_and_bad_yaml(tmp_path: Path) -> None:
    (tmp_path / "list.yaml").write_text("- 1\n- 2\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="mapping"):
        load_suite(tmp_path / "list.yaml")
    (tmp_path / "bad.yaml").write_text("a: [1,\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="invalid YAML"):
        load_suite(tmp_path / "bad.yaml")
    with pytest.raises(ConfigError, match="not found"):
        load_suite(tmp_path / "missing.yaml")
