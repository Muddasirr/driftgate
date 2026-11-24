# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [0.2.0] - 2026-11-24

### Added
- `anthropic` provider for Claude models, built on the official Anthropic SDK (`pip install "driftgate[anthropic]"`). Cases need no changes: system prompts, tool schemas, tool calls and tool results are translated to the Messages API. Per-provider `max_tokens` (thinking counts toward it) and `effort`.
- Claude Opus 5.5 in the voice-intake case study.

### Changed
- OpenAI-compatible provider now shares its request loop through an `HTTPProvider` base class. Behaviour is unchanged, but cache keys now include the full endpoint URL, so the first run after upgrading misses the cache.

### Notes
- No sampling parameters are sent to Claude (current models reject `temperature`), and refusal fallbacks are deliberately not enabled: a refusal is recorded as a failed attempt so results always describe the model named in the suite.

## [0.1.0] - 2026-11-24

### Added
- YAML suite format with up-front validation (pydantic), `file://` references for prompts and tools.
- OpenAI-compatible provider with retries, exponential backoff, `Retry-After` support and per-call cost; mock provider for offline runs.
- Assertions: `contains`, `not_contains`, `equals`, `regex`, `json_valid`, `json_schema`, `tool_call`, `no_tool_call`, `max_latency_ms`, `max_cost_usd`, `llm_judge`.
- Conversation history with assistant tool calls and tool results.
- SQLite response cache.
- Baseline comparison (regressed / fixed / still failing / new / removed) and gate thresholds for regressions, pass rate, cost and p95 latency.
- CLI: `run`, `compare`, `validate`, `init`; `--concurrency` override for rate-limited keys.
- Markdown report, GitHub job summary and a composite GitHub Action that comments on pull requests.
- Examples: offline quickstart and a voice-agent case study.
