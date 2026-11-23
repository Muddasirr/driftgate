"""Human-readable output: a Markdown report (for PR comments and job summaries)
and a compact console summary."""

from __future__ import annotations

from driftgate.compare import CaseDiff, Comparison, ProviderDelta, Status

COMMENT_MARKER = "<!-- driftgate-report -->"


def markdown(comparison: Comparison) -> str:
    verdict = "✅ passed" if comparison.passed else "❌ failed"
    regressions = comparison.by_status(Status.REGRESSED)
    fixed = comparison.by_status(Status.FIXED)
    failing = comparison.by_status(Status.STILL_FAILING, Status.NEW_FAILING)

    lines = [COMMENT_MARKER, f"## driftgate · {comparison.suite} · {verdict}", ""]
    if comparison.gate_failures:
        lines += [f"- {f}" for f in comparison.gate_failures] + [""]

    lines += [
        "| Provider | Pass rate | Cost | p95 latency |",
        "|---|---|---|---|",
        *(_provider_row(p) for p in comparison.providers),
        "",
    ]

    if regressions:
        lines += _case_table(f"Regressions ({len(regressions)})", regressions)
    if fixed:
        lines += [f"**Fixed ({len(fixed)}):** " + ", ".join(f"`{_label(d)}`" for d in fixed), ""]
    if failing:
        lines += [
            f"<details><summary>Failing, not new ({len(failing)})</summary>",
            "",
            *_case_table(None, failing),
            "</details>",
            "",
        ]

    if comparison.has_baseline:
        base = (comparison.baseline_sha or "unknown")[:7]
        lines.append(f"<sub>Compared against baseline `{base}`.</sub>")
    else:
        lines.append("<sub>No baseline yet: this run will become the baseline.</sub>")
    return "\n".join(lines) + "\n"


def console(comparison: Comparison) -> str:
    lines = []
    for p in comparison.providers:
        s = p.current
        lines.append(
            f"{p.provider_id}: {s.passed}/{s.total} passed ({s.pass_rate:.0%})"
            f"{_delta_pp(p.pass_rate_delta)}, cost {_money(s.cost_usd)}, "
            f"p95 {_ms(s.p95_latency_ms)}"
        )
    for d in comparison.by_status(Status.REGRESSED):
        lines.append(f"  REGRESSED  {_label(d)}  {d.failure_detail}")
    for d in comparison.by_status(Status.STILL_FAILING, Status.NEW_FAILING):
        lines.append(f"  FAILING    {_label(d)}  {d.failure_detail}")
    for d in comparison.by_status(Status.FIXED):
        lines.append(f"  FIXED      {_label(d)}")
    if comparison.gate_failures:
        lines.append("GATE FAILED: " + "; ".join(comparison.gate_failures))
    else:
        lines.append("Gate passed.")
    return "\n".join(lines)


def _provider_row(p: ProviderDelta) -> str:
    s = p.current
    rate = f"{s.passed}/{s.total} ({s.pass_rate:.0%}){_delta_pp(p.pass_rate_delta)}"
    cost = _money(s.cost_usd) + _delta_pct(p.cost_change_pct)
    p95 = _ms(s.p95_latency_ms) + _delta_pct(p.p95_change_pct)
    return f"| `{p.provider_id}` | {rate} | {cost} | {p95} |"


def _case_table(title: str | None, diffs: list[CaseDiff]) -> list[str]:
    head = [f"### {title}", ""] if title else []
    rows = [f"| `{d.case_id}` | `{d.provider_id}` | {_escape(d.failure_detail)} |" for d in diffs]
    return [*head, "| Case | Provider | First failure |", "|---|---|---|", *rows, ""]


def _label(d: CaseDiff) -> str:
    return f"{d.provider_id}/{d.case_id}"


def _escape(text: str, limit: int = 200) -> str:
    text = text.replace("|", "\\|").replace("\n", " ")
    return text if len(text) <= limit else text[:limit] + "…"


def _money(value: float | None) -> str:
    return "n/a" if value is None else f"${value:.4f}"


def _ms(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0f} ms"


def _delta_pp(delta: float | None) -> str:
    if delta is None or abs(delta) < 1e-9:
        return ""
    return f" ({delta * 100:+.0f} pp)"


def _delta_pct(delta: float | None) -> str:
    if delta is None or abs(delta) < 0.05:
        return ""
    return f" ({delta:+.1f}%)"
