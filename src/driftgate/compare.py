"""Baseline comparison and the gate decision."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from driftgate.config import Thresholds
from driftgate.results import CaseResult, RunResult, Summary


class Status(StrEnum):
    REGRESSED = "regressed"
    FIXED = "fixed"
    STILL_FAILING = "still_failing"
    PASSING = "passing"
    NEW_PASSING = "new_passing"
    NEW_FAILING = "new_failing"
    REMOVED = "removed"


@dataclass(frozen=True)
class CaseDiff:
    provider_id: str
    case_id: str
    status: Status
    current: CaseResult | None
    baseline: CaseResult | None

    @property
    def failure_detail(self) -> str:
        """First failing assertion (or error) of the current run, for reports."""
        if self.current is None:
            return ""
        for attempt in self.current.attempts:
            if attempt.error:
                return f"error: {attempt.error}"
            for outcome in attempt.assertions:
                if not outcome.passed:
                    return f"{outcome.type}: {outcome.detail}".rstrip(": ")
        return ""


@dataclass(frozen=True)
class ProviderDelta:
    provider_id: str
    current: Summary
    baseline: Summary | None

    @property
    def pass_rate_delta(self) -> float | None:
        return None if self.baseline is None else self.current.pass_rate - self.baseline.pass_rate

    @property
    def cost_change_pct(self) -> float | None:
        return _pct_change(self.baseline.cost_usd if self.baseline else None, self.current.cost_usd)

    @property
    def p95_change_pct(self) -> float | None:
        return _pct_change(
            self.baseline.p95_latency_ms if self.baseline else None, self.current.p95_latency_ms
        )


@dataclass
class Comparison:
    suite: str
    diffs: list[CaseDiff]
    providers: list[ProviderDelta]
    baseline_sha: str | None
    current_sha: str | None
    has_baseline: bool
    gate_failures: list[str] = field(default_factory=list)

    def by_status(self, *statuses: Status) -> list[CaseDiff]:
        return [d for d in self.diffs if d.status in statuses]

    @property
    def passed(self) -> bool:
        return not self.gate_failures


def compare(
    current: RunResult, baseline: RunResult | None, thresholds: Thresholds | None = None
) -> Comparison:
    thresholds = thresholds or Thresholds()
    cur = current.by_key()
    base = baseline.by_key() if baseline else {}

    diffs = []
    for key in list(cur) + [k for k in base if k not in cur]:
        diffs.append(_diff(key, cur.get(key), base.get(key), has_baseline=baseline is not None))
    diffs.sort(key=lambda d: (d.provider_id, d.case_id))

    deltas = []
    for pid in current.provider_ids():
        in_baseline = baseline is not None and pid in baseline.provider_ids()
        deltas.append(
            ProviderDelta(
                provider_id=pid,
                current=current.summary(pid),
                baseline=baseline.summary(pid) if baseline and in_baseline else None,
            )
        )

    comparison = Comparison(
        suite=current.suite,
        diffs=diffs,
        providers=deltas,
        baseline_sha=baseline.git_sha if baseline else None,
        current_sha=current.git_sha,
        has_baseline=baseline is not None,
    )
    comparison.gate_failures = _gate(comparison, thresholds)
    return comparison


def _diff(
    key: tuple[str, str],
    current: CaseResult | None,
    baseline: CaseResult | None,
    *,
    has_baseline: bool,
) -> CaseDiff:
    provider_id, case_id = key
    if current is None:
        status = Status.REMOVED
    elif baseline is None:
        if has_baseline:
            status = Status.NEW_PASSING if current.passed else Status.NEW_FAILING
        else:
            status = Status.PASSING if current.passed else Status.STILL_FAILING
    elif baseline.passed and not current.passed:
        status = Status.REGRESSED
    elif not baseline.passed and current.passed:
        status = Status.FIXED
    elif current.passed:
        status = Status.PASSING
    else:
        status = Status.STILL_FAILING
    return CaseDiff(provider_id, case_id, status, current, baseline)


def _gate(comparison: Comparison, t: Thresholds) -> list[str]:
    failures = []
    regressions = comparison.by_status(Status.REGRESSED)
    if comparison.has_baseline and len(regressions) > t.max_regressions:
        failures.append(f"{len(regressions)} regression(s), max allowed {t.max_regressions}")
    for delta in comparison.providers:
        prefix = f"[{delta.provider_id}]"
        if t.min_pass_rate is not None and delta.current.pass_rate < t.min_pass_rate:
            failures.append(
                f"{prefix} pass rate {delta.current.pass_rate:.0%} "
                f"below minimum {t.min_pass_rate:.0%}"
            )
        cost = delta.cost_change_pct
        if (
            t.max_cost_increase_pct is not None
            and cost is not None
            and cost > t.max_cost_increase_pct
        ):
            failures.append(
                f"{prefix} cost up {cost:.1f}%, max allowed {t.max_cost_increase_pct:.1f}%"
            )
        p95 = delta.p95_change_pct
        limit = t.max_p95_latency_increase_pct
        if limit is not None and p95 is not None and p95 > limit:
            failures.append(f"{prefix} p95 latency up {p95:.1f}%, max allowed {limit:.1f}%")
    return failures


def _pct_change(before: float | None, after: float | None) -> float | None:
    if before is None or after is None or before == 0:
        return None
    return (after - before) / before * 100
