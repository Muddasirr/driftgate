# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

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
