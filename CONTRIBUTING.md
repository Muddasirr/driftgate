# Contributing

Thanks for helping. Issues and pull requests are welcome.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Before opening a pull request

```bash
ruff check . && ruff format --check .
mypy
pytest --cov=driftgate
```

CI runs the same checks on Python 3.11, 3.12 and 3.13 and requires 90% coverage.
Tests must not call real model APIs: use `respx` for HTTP and the `mock` provider
for end-to-end runs.

## Adding an assertion

1. Add a model to `src/driftgate/config.py` and include it in the `Assertion` union.
2. Handle it in `evaluate()` in `src/driftgate/assertions.py`.
3. Add tests in `tests/test_assertions.py` and a row in the README's assertion table.

## Commit messages

Write the subject in the imperative ("Add regex args to tool_call"), and use the
body to explain why the change is needed.
