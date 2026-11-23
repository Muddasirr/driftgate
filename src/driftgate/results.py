"""Result types. A run serialises to JSON, and that same file is the baseline for
the next run, so the schema is versioned."""

from __future__ import annotations

import math
from pathlib import Path

from pydantic import BaseModel, Field

from driftgate.config import ToolCall

SCHEMA_VERSION = 1


class Completion(BaseModel):
    content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    cost_usd: float | None = None
    cached: bool = False


class AssertionOutcome(BaseModel):
    type: str
    passed: bool
    detail: str = ""


class Attempt(BaseModel):
    completion: Completion | None = None
    error: str | None = None
    assertions: list[AssertionOutcome] = Field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.error is None and all(a.passed for a in self.assertions)


class CaseResult(BaseModel):
    case_id: str
    provider_id: str
    attempts: list[Attempt]

    @property
    def passed(self) -> bool:
        """A case passes only if every repeat passed, so flaky behaviour fails loudly."""
        return all(a.passed for a in self.attempts)

    @property
    def pass_count(self) -> int:
        return sum(a.passed for a in self.attempts)

    @property
    def latency_ms(self) -> float | None:
        values = [
            a.completion.latency_ms
            for a in self.attempts
            if a.completion and not a.completion.cached
        ]
        return _percentile(values, 50) if values else None

    @property
    def cost_usd(self) -> float | None:
        costs = [a.completion.cost_usd for a in self.attempts if a.completion]
        known = [c for c in costs if c is not None]
        return sum(known) if known else None

    @property
    def errored(self) -> bool:
        return any(a.error for a in self.attempts)


class Summary(BaseModel):
    total: int
    passed: int
    errors: int
    pass_rate: float
    cost_usd: float | None
    p50_latency_ms: float | None
    p95_latency_ms: float | None


class RunResult(BaseModel):
    schema_version: int = SCHEMA_VERSION
    suite: str
    driftgate_version: str
    started_at: str
    finished_at: str
    git_sha: str | None = None
    results: list[CaseResult]

    def provider_ids(self) -> list[str]:
        return list(dict.fromkeys(r.provider_id for r in self.results))

    def by_key(self) -> dict[tuple[str, str], CaseResult]:
        return {(r.provider_id, r.case_id): r for r in self.results}

    def summary(self, provider_id: str | None = None) -> Summary:
        rows = [r for r in self.results if provider_id is None or r.provider_id == provider_id]
        passed = sum(r.passed for r in rows)
        latencies = [r.latency_ms for r in rows if r.latency_ms is not None]
        costs = [r.cost_usd for r in rows if r.cost_usd is not None]
        return Summary(
            total=len(rows),
            passed=passed,
            errors=sum(r.errored for r in rows),
            pass_rate=passed / len(rows) if rows else 0.0,
            cost_usd=sum(costs) if costs else None,
            p50_latency_ms=_percentile(latencies, 50) if latencies else None,
            p95_latency_ms=_percentile(latencies, 95) if latencies else None,
        )

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(self.model_dump_json(indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> RunResult:
        data = Path(path).read_text(encoding="utf-8")
        result = cls.model_validate_json(data)
        if result.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"{path}: result schema v{result.schema_version} is not supported "
                f"(expected v{SCHEMA_VERSION})"
            )
        return result


def _percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile; matches what most dashboards report for small samples."""
    ordered = sorted(values)
    rank = max(1, math.ceil(pct / 100 * len(ordered)))
    return ordered[rank - 1]
