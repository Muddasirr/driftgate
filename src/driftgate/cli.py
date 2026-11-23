"""Command-line interface.

Exit codes: 0 gate passed · 1 gate failed · 2 invalid input or config.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import click

from driftgate import __version__
from driftgate.cache import DEFAULT_CACHE_PATH, ResponseCache
from driftgate.compare import Comparison, compare
from driftgate.config import ConfigError, Suite, Thresholds, load_suite
from driftgate.providers import ProviderError
from driftgate.report import console, markdown
from driftgate.results import CaseResult, RunResult
from driftgate.runner import run_suite

EXIT_GATE_FAILED = 1
EXIT_USAGE = 2

EXAMPLE_SUITE = """\
# driftgate suite: https://github.com/Muddasirr/driftgate
name: support-bot
system: |
  You are a concise support assistant for Acme. Answer in at most two sentences.
  If you don't know, say so. Never promise refunds.

providers:
  - id: gpt-4o-mini
    type: openai
    model: gpt-4o-mini
    api_key_env: OPENAI_API_KEY
    pricing: {input_per_mtok: 0.15, output_per_mtok: 0.60}

judge: gpt-4o-mini

thresholds:
  min_pass_rate: 0.9
  max_regressions: 0

cases:
  - id: refund-policy-no-promise
    input: My order arrived broken, I want my money back now.
    assert:
      - type: not_contains
        value: "I will refund"
      - type: llm_judge
        rubric: Acknowledges the problem and offers a next step without promising a refund.

  - id: unknown-says-so
    input: What's the Wi-Fi password at your Berlin office?
    assert:
      - type: llm_judge
        rubric: Says it doesn't know or can't share this, without inventing a password.
"""


def _fail(message: str) -> None:
    click.secho(message, fg="red", err=True)
    sys.exit(EXIT_USAGE)


def _load(path: str) -> Suite:
    try:
        return load_suite(path)
    except ConfigError as exc:
        _fail(str(exc))
        raise  # unreachable; keeps type checkers happy


def _print_progress(result: CaseResult) -> None:
    status = click.style("PASS", fg="green") if result.passed else click.style("FAIL", fg="red")
    repeats = f" ({result.pass_count}/{len(result.attempts)})" if len(result.attempts) > 1 else ""
    click.echo(f"  {status} {result.provider_id}/{result.case_id}{repeats}", err=True)


def _emit(comparison: Comparison, markdown_path: str | None) -> None:
    click.echo(console(comparison))
    md = markdown(comparison)
    if markdown_path:
        Path(markdown_path).parent.mkdir(parents=True, exist_ok=True)
        Path(markdown_path).write_text(md, encoding="utf-8")
    if summary := os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(summary, "a", encoding="utf-8") as fh:
            fh.write(md)


@click.group()
@click.version_option(__version__, prog_name="driftgate")
def main() -> None:
    """Catch LLM quality drift before it merges."""


@main.command()
@click.argument("suite_path", metavar="SUITE", type=click.Path(dir_okay=False))
@click.option("--provider", "providers", multiple=True, help="Only run these provider ids.")
@click.option("--tag", "tags", multiple=True, help="Only run cases with any of these tags.")
@click.option("--case", "cases", multiple=True, help="Only run these case ids.")
@click.option(
    "--out",
    default=".driftgate/results.json",
    show_default=True,
    type=click.Path(dir_okay=False),
    help="Where to write the run's JSON results.",
)
@click.option(
    "--baseline",
    type=click.Path(dir_okay=False),
    help="Results JSON from a previous run to compare against. Skipped if the file is missing.",
)
@click.option(
    "--markdown", "markdown_path", type=click.Path(dir_okay=False), help="Write a Markdown report."
)
@click.option("--no-cache", is_flag=True, help="Always call the model, ignoring cached responses.")
@click.option(
    "--cache-path",
    default=str(DEFAULT_CACHE_PATH),
    show_default=True,
    type=click.Path(dir_okay=False),
)
def run(
    suite_path: str,
    providers: tuple[str, ...],
    tags: tuple[str, ...],
    cases: tuple[str, ...],
    out: str,
    baseline: str | None,
    markdown_path: str | None,
    no_cache: bool,
    cache_path: str,
) -> None:
    """Run SUITE, compare with --baseline if given, and exit 1 if the gate fails."""
    suite = _load(suite_path)
    cache = None if no_cache else ResponseCache(cache_path)
    try:
        result = asyncio.run(
            run_suite(
                suite,
                provider_ids=providers,
                tags=tags,
                case_ids=cases,
                cache=cache,
                on_result=_print_progress,
            )
        )
    except (ProviderError, KeyError, ValueError) as exc:
        _fail(f"error: {exc}")
        return
    finally:
        if cache is not None:
            cache.close()

    result.save(out)
    base = _load_baseline(baseline)
    comparison = compare(result, base, suite.thresholds)
    _emit(comparison, markdown_path)
    sys.exit(0 if comparison.passed else EXIT_GATE_FAILED)


@main.command(name="compare")
@click.argument("baseline", type=click.Path(exists=True, dir_okay=False))
@click.argument("current", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--suite",
    "suite_path",
    type=click.Path(dir_okay=False),
    help="Read thresholds from this suite.",
)
@click.option(
    "--markdown", "markdown_path", type=click.Path(dir_okay=False), help="Write a Markdown report."
)
def compare_cmd(
    baseline: str, current: str, suite_path: str | None, markdown_path: str | None
) -> None:
    """Compare two results files without calling any model."""
    thresholds = _load(suite_path).thresholds if suite_path else Thresholds()
    try:
        comparison = compare(RunResult.load(current), RunResult.load(baseline), thresholds)
    except ValueError as exc:
        _fail(str(exc))
        return
    _emit(comparison, markdown_path)
    sys.exit(0 if comparison.passed else EXIT_GATE_FAILED)


@main.command()
@click.argument("suite_path", metavar="SUITE", type=click.Path(dir_okay=False))
def validate(suite_path: str) -> None:
    """Check that SUITE is valid without running it."""
    suite = _load(suite_path)
    n_assertions = sum(len(c.assertions) for c in suite.cases)
    click.echo(
        f"{suite_path}: OK · {len(suite.cases)} cases, {n_assertions} assertions, "
        f"{len(suite.providers)} provider(s)"
    )


@main.command()
@click.argument("path", default="driftgate.yaml", type=click.Path(dir_okay=False))
def init(path: str) -> None:
    """Write an example suite to PATH."""
    target = Path(path)
    if target.exists():
        _fail(f"{target} already exists")
    target.write_text(EXAMPLE_SUITE, encoding="utf-8")
    click.echo(f"Wrote {target}. Set OPENAI_API_KEY, then run: driftgate run {target}")


def _load_baseline(path: str | None) -> RunResult | None:
    if not path:
        return None
    if not Path(path).is_file():
        click.echo(f"No baseline at {path}; this run has nothing to compare against.", err=True)
        return None
    try:
        return RunResult.load(path)
    except ValueError as exc:
        _fail(f"baseline {path}: {exc}")
        return None
