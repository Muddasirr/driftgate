"""Runs a suite: every (provider, case, repeat) concurrently, bounded by
`defaults.concurrency`, then evaluates each case's assertions."""

from __future__ import annotations

import asyncio
import os
import subprocess
from collections.abc import Callable, Iterable, Mapping
from datetime import UTC, datetime

from driftgate import __version__
from driftgate.assertions import JudgeContext, evaluate
from driftgate.cache import ResponseCache
from driftgate.config import Case, LLMJudgeAssertion, Suite
from driftgate.providers import Provider, ProviderError, Request, build_provider
from driftgate.results import Attempt, CaseResult, RunResult

ProgressCallback = Callable[[CaseResult], None]


def build_messages(suite: Suite, case: Case) -> list[dict[str, str]]:
    if case.messages is not None:
        messages = [m.model_dump() for m in case.messages]
        if suite.system and not any(m["role"] == "system" for m in messages):
            messages.insert(0, {"role": "system", "content": suite.system})
        return messages
    messages = [{"role": "user", "content": case.input or ""}]
    if suite.system:
        messages.insert(0, {"role": "system", "content": suite.system})
    return messages


def _conversation_text(messages: list[dict[str, str]]) -> str:
    # The system prompt is left out: the judge grades behaviour, not the prompt.
    return "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in messages if m["role"] != "system"
    )


def select_cases(suite: Suite, tags: Iterable[str] = (), ids: Iterable[str] = ()) -> list[Case]:
    tag_set, id_set = set(tags), set(ids)
    cases = suite.cases
    if tag_set:
        cases = [c for c in cases if tag_set & set(c.tags)]
    if id_set:
        unknown = id_set - {c.id for c in suite.cases}
        if unknown:
            raise ValueError(f"unknown case id(s): {', '.join(sorted(unknown))}")
        cases = [c for c in cases if c.id in id_set]
    return cases


def _judge_ids(suite: Suite, cases: Iterable[Case]) -> set[str]:
    ids = set()
    for case in cases:
        for a in case.assertions:
            if isinstance(a, LLMJudgeAssertion):
                judge = a.provider or suite.judge
                if judge:
                    ids.add(judge)
    return ids


async def run_suite(
    suite: Suite,
    *,
    provider_ids: Iterable[str] = (),
    tags: Iterable[str] = (),
    case_ids: Iterable[str] = (),
    cache: ResponseCache | None = None,
    providers: Mapping[str, Provider] | None = None,
    on_result: ProgressCallback | None = None,
) -> RunResult:
    cases = select_cases(suite, tags, case_ids)
    targets = list(provider_ids) or [p.id for p in suite.providers]
    for pid in targets:
        suite.provider(pid)  # raises KeyError for unknown ids

    needed = set(targets) | _judge_ids(suite, cases)
    owned: dict[str, Provider] = {}
    if providers is None:
        try:
            for pid in sorted(needed):
                owned[pid] = build_provider(suite.provider(pid), cache=cache)
        except ProviderError:
            await asyncio.gather(*(p.aclose() for p in owned.values()))
            raise
        providers = owned

    started_at = _now()
    semaphore = asyncio.Semaphore(suite.defaults.concurrency)
    try:
        jobs = [
            _run_case(suite, case, providers[pid], providers, semaphore, on_result)
            for pid in targets
            for case in cases
        ]
        results = await asyncio.gather(*jobs)
    finally:
        await asyncio.gather(*(p.aclose() for p in owned.values()))

    return RunResult(
        suite=suite.name,
        driftgate_version=__version__,
        started_at=started_at,
        finished_at=_now(),
        git_sha=_git_sha(),
        results=list(results),
    )


async def _run_case(
    suite: Suite,
    case: Case,
    provider: Provider,
    all_providers: Mapping[str, Provider],
    semaphore: asyncio.Semaphore,
    on_result: ProgressCallback | None,
) -> CaseResult:
    messages = build_messages(suite, case)
    repeat = case.repeat or suite.defaults.repeat
    judge = JudgeContext(
        providers=all_providers,
        default_judge=suite.judge,
        case_id=case.id,
        conversation=_conversation_text(messages),
    )

    async def one(attempt_index: int) -> Attempt:
        request = Request(
            case_id=case.id,
            messages=messages,
            tools=suite.tools,
            temperature=suite.defaults.temperature,
            max_tokens=suite.defaults.max_tokens,
            attempt=attempt_index,
        )
        async with semaphore:
            try:
                completion = await provider.complete(request)
            except ProviderError as exc:
                return Attempt(error=str(exc))
        outcomes = [await evaluate(a, completion, judge) for a in case.assertions]
        return Attempt(completion=completion, assertions=outcomes)

    attempts = await asyncio.gather(*(one(i) for i in range(repeat)))
    result = CaseResult(case_id=case.id, provider_id=provider.id, attempts=list(attempts))
    if on_result is not None:
        on_result(result)
    return result


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _git_sha() -> str | None:
    if sha := os.environ.get("GITHUB_SHA"):
        return sha
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5, check=True
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None
