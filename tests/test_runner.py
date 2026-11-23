from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from driftgate.config import load_suite
from driftgate.providers import Provider, ProviderError, Request
from driftgate.results import Completion
from driftgate.runner import build_messages, run_suite, select_cases
from tests.conftest import mock_suite, write_suite


class Recorder:
    """Provider double that records requests and tracks peak concurrency."""

    def __init__(self, pid: str, reply: str = "ok", fail: bool = False) -> None:
        self.id = pid
        self.reply = reply
        self.fail = fail
        self.requests: list[Request] = []
        self.in_flight = 0
        self.peak = 0

    async def complete(self, request: Request) -> Completion:
        self.requests.append(request)
        self.in_flight += 1
        self.peak = max(self.peak, self.in_flight)
        await asyncio.sleep(0.01)
        self.in_flight -= 1
        if self.fail:
            raise ProviderError("down")
        return Completion(content=self.reply, latency_ms=5)

    async def aclose(self) -> None:
        return None


def test_build_messages_prepends_system(tmp_path: Path) -> None:
    data = mock_suite(
        {},
        [
            {"id": "a", "input": "hi", "assert": [{"type": "json_valid"}]},
            {
                "id": "b",
                "messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hello"},
                    {"role": "user", "content": "and?"},
                ],
                "assert": [{"type": "json_valid"}],
            },
        ],
        system="Be brief.",
    )
    suite = load_suite(write_suite(tmp_path, data))
    assert build_messages(suite, suite.cases[0]) == [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "hi"},
    ]
    assert [m["role"] for m in build_messages(suite, suite.cases[1])] == [
        "system",
        "user",
        "assistant",
        "user",
    ]


async def test_concurrency_limit_repeats_and_judge(tmp_path: Path) -> None:
    cases = [
        {
            "id": f"c{i}",
            "input": f"q{i}",
            "repeat": 2,
            "assert": [
                {"type": "contains", "value": "ok"},
                {"type": "llm_judge", "rubric": "fine"},
            ],
        }
        for i in range(6)
    ]
    data = mock_suite({}, cases, judge="judge", defaults={"concurrency": 2})
    data["providers"].append({"id": "judge", "type": "mock"})
    suite = load_suite(write_suite(tmp_path, data))

    target = Recorder("mock")
    judge = Recorder("judge", reply='{"score": 5, "reason": "good"}')
    providers: dict[str, Provider] = {"mock": target, "judge": judge}
    result = await run_suite(suite, provider_ids=["mock"], providers=providers)

    assert len(result.results) == 6 and all(r.passed for r in result.results)
    assert len(target.requests) == 12  # 6 cases x 2 repeats
    assert target.peak <= 2
    assert {r.attempt for r in target.requests} == {0, 1}
    assert len(judge.requests) == 12


async def test_provider_errors_become_failed_attempts(tmp_path: Path) -> None:
    data = mock_suite({}, [{"id": "a", "input": "x", "assert": [{"type": "json_valid"}]}])
    suite = load_suite(write_suite(tmp_path, data))
    result = await run_suite(suite, providers={"mock": Recorder("mock", fail=True)})
    row = result.results[0]
    assert not row.passed and row.errored
    assert row.attempts[0].error == "down"


def test_select_cases(tmp_path: Path) -> None:
    cases = [
        {"id": "a", "input": "x", "tags": ["smoke"], "assert": [{"type": "json_valid"}]},
        {"id": "b", "input": "x", "assert": [{"type": "json_valid"}]},
    ]
    suite = load_suite(write_suite(tmp_path, mock_suite({}, cases)))
    assert [c.id for c in select_cases(suite, tags=["smoke"])] == ["a"]
    assert [c.id for c in select_cases(suite, ids=["b"])] == ["b"]
    with pytest.raises(ValueError, match="unknown case id"):
        select_cases(suite, ids=["zzz"])


def test_tool_call_history_uses_openai_wire_format(tmp_path: Path) -> None:
    case = {
        "id": "returning",
        "messages": [
            {"role": "user", "content": "I'm Maria, 512 555 0199"},
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "call_1", "name": "lookup", "arguments": {"phone": "5125550199"}}
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": '{"found": true}'},
        ],
        "assert": [{"type": "json_valid"}],
    }
    suite = load_suite(write_suite(tmp_path, mock_suite({}, [case])))
    wire = build_messages(suite, suite.cases[0])
    assert wire[1] == {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "lookup", "arguments": '{"phone": "5125550199"}'},
            }
        ],
    }
    assert wire[2] == {"role": "tool", "tool_call_id": "call_1", "content": '{"found": true}'}

    from driftgate.runner import _conversation_text

    transcript = _conversation_text(wire)
    assert 'ASSISTANT CALLED lookup({"phone": "5125550199"})' in transcript
    assert 'TOOL RESULT: {"found": true}' in transcript


@pytest.mark.parametrize(
    ("message", "error"),
    [
        ({"role": "tool", "content": "x"}, "tool_call_id"),
        ({"role": "user", "tool_call_id": "c", "content": "x"}, "only allowed on tool"),
        (
            {"role": "user", "content": "x", "tool_calls": [{"id": "c", "name": "f"}]},
            "only allowed on assistant",
        ),
        ({"role": "assistant"}, "'content' or 'tool_calls'"),
        ({"role": "user"}, "need 'content'"),
    ],
)
def test_rejects_malformed_history(tmp_path: Path, message: dict[str, object], error: str) -> None:
    from driftgate.config import ConfigError

    case = {"id": "a", "messages": [message], "assert": [{"type": "json_valid"}]}
    with pytest.raises(ConfigError, match=error):
        load_suite(write_suite(tmp_path, mock_suite({}, [case])))
