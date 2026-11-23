"""Suite file schema and loader.

A suite is a YAML file describing which models to call, what to send them and
what a good answer looks like. Everything is validated up front so a typo fails
in milliseconds instead of after a paid run.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

FILE_REF = "file://"
_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


class ConfigError(Exception):
    """Raised when a suite file can't be loaded or is invalid."""


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# --------------------------------------------------------------------- providers


class Pricing(_Model):
    """USD per million tokens, used to compute the cost of each call."""

    input_per_mtok: float = Field(ge=0)
    output_per_mtok: float = Field(ge=0)


class ToolCall(_Model):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class MockResponse(_Model):
    content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0


class OpenAIProviderSpec(_Model):
    """Any OpenAI-compatible chat completions API: OpenAI, Groq, Together, vLLM, Ollama."""

    type: Literal["openai"]
    id: str
    model: str
    base_url: str = "https://api.openai.com/v1"
    api_key_env: str = "OPENAI_API_KEY"
    pricing: Pricing | None = None
    timeout_s: float = Field(default=60.0, gt=0)
    max_retries: int = Field(default=4, ge=0, le=10)


class MockProviderSpec(_Model):
    """Canned responses keyed by case id. For tests, demos and offline CI."""

    type: Literal["mock"]
    id: str
    responses: dict[str, MockResponse] = Field(default_factory=dict)
    default: MockResponse | None = None
    latency_ms: float = Field(default=0.0, ge=0)
    pricing: Pricing | None = None


ProviderSpec = Annotated[OpenAIProviderSpec | MockProviderSpec, Field(discriminator="type")]


# -------------------------------------------------------------------- assertions


class ContainsAssertion(_Model):
    type: Literal["contains"]
    value: str
    case_sensitive: bool = False


class NotContainsAssertion(_Model):
    type: Literal["not_contains"]
    value: str
    case_sensitive: bool = False


class EqualsAssertion(_Model):
    type: Literal["equals"]
    value: str
    strip: bool = True


class RegexAssertion(_Model):
    type: Literal["regex"]
    pattern: str

    @model_validator(mode="after")
    def _compiles(self) -> RegexAssertion:
        try:
            re.compile(self.pattern)
        except re.error as exc:
            raise ValueError(f"invalid regex {self.pattern!r}: {exc}") from exc
        return self


class JsonValidAssertion(_Model):
    type: Literal["json_valid"]


class JsonSchemaAssertion(_Model):
    type: Literal["json_schema"]
    json_schema: dict[str, Any] = Field(alias="schema")

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)


class ToolCallAssertion(_Model):
    """The model must call `name`. `args` values match exactly, or by {regex: ...}."""

    type: Literal["tool_call"]
    name: str
    args: dict[str, Any] = Field(default_factory=dict)
    position: Literal["any", "first"] = "any"


class NoToolCallAssertion(_Model):
    """The model must not call `name` (or any tool, if `name` is omitted)."""

    type: Literal["no_tool_call"]
    name: str | None = None


class MaxLatencyAssertion(_Model):
    type: Literal["max_latency_ms"]
    value: float = Field(gt=0)


class MaxCostAssertion(_Model):
    type: Literal["max_cost_usd"]
    value: float = Field(gt=0)


class LLMJudgeAssertion(_Model):
    """Ask a judge model to score the output 1-5 against a rubric."""

    type: Literal["llm_judge"]
    rubric: str
    min_score: int = Field(default=4, ge=1, le=5)
    provider: str | None = None


Assertion = Annotated[
    ContainsAssertion
    | NotContainsAssertion
    | EqualsAssertion
    | RegexAssertion
    | JsonValidAssertion
    | JsonSchemaAssertion
    | ToolCallAssertion
    | NoToolCallAssertion
    | MaxLatencyAssertion
    | MaxCostAssertion
    | LLMJudgeAssertion,
    Field(discriminator="type"),
]


# ------------------------------------------------------------------- suite + cases


class HistoryToolCall(_Model):
    """A tool call the assistant made earlier in the conversation."""

    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class Message(_Model):
    """One conversation turn. Assistant turns may carry `tool_calls`; `tool` turns
    carry the result for a `tool_call_id`, so agent histories can be replayed."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    tool_calls: list[HistoryToolCall] = Field(default_factory=list)
    tool_call_id: str | None = None

    @model_validator(mode="after")
    def _shape(self) -> Message:
        if self.role == "tool":
            if not self.tool_call_id or self.content is None:
                raise ValueError("tool messages need 'tool_call_id' and 'content'")
        elif self.tool_call_id is not None:
            raise ValueError("'tool_call_id' is only allowed on tool messages")
        if self.tool_calls and self.role != "assistant":
            raise ValueError("'tool_calls' is only allowed on assistant messages")
        if self.role in ("system", "user") and self.content is None:
            raise ValueError(f"{self.role} messages need 'content'")
        if self.role == "assistant" and self.content is None and not self.tool_calls:
            raise ValueError("assistant messages need 'content' or 'tool_calls'")
        return self

    def to_wire(self) -> dict[str, Any]:
        """OpenAI chat-completions format."""
        if self.role == "tool":
            return {"role": "tool", "tool_call_id": self.tool_call_id, "content": self.content}
        message: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            message["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
                }
                for call in self.tool_calls
            ]
        return message


class Case(_Model):
    id: str
    description: str | None = None
    input: str | None = None
    messages: list[Message] | None = None
    assertions: list[Assertion] = Field(alias="assert", min_length=1)
    tags: list[str] = Field(default_factory=list)
    repeat: int | None = Field(default=None, ge=1, le=20)

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    @model_validator(mode="after")
    def _one_input(self) -> Case:
        if not _ID_PATTERN.match(self.id):
            raise ValueError(
                f"case id {self.id!r} must be lowercase letters, digits, '.', '_' or '-'"
            )
        if (self.input is None) == (self.messages is None):
            raise ValueError(f"case {self.id!r}: set exactly one of 'input' or 'messages'")
        return self


class Defaults(_Model):
    temperature: float = Field(default=0.0, ge=0, le=2)
    max_tokens: int = Field(default=512, ge=1)
    repeat: int = Field(default=1, ge=1, le=20)
    concurrency: int = Field(default=4, ge=1, le=64)


class Thresholds(_Model):
    """Gate conditions. A run fails if any configured threshold is breached."""

    min_pass_rate: float | None = Field(default=None, ge=0, le=1)
    max_regressions: int = Field(default=0, ge=0)
    max_cost_increase_pct: float | None = Field(default=None, ge=0)
    max_p95_latency_increase_pct: float | None = Field(default=None, ge=0)


class Suite(_Model):
    name: str
    description: str | None = None
    system: str | None = None
    tools: list[dict[str, Any]] | None = None
    providers: list[ProviderSpec] = Field(min_length=1)
    judge: str | None = None
    defaults: Defaults = Field(default_factory=Defaults)
    thresholds: Thresholds = Field(default_factory=Thresholds)
    cases: list[Case] = Field(min_length=1)

    @model_validator(mode="after")
    def _references(self) -> Suite:
        provider_ids = [p.id for p in self.providers]
        _require_unique("provider id", provider_ids)
        _require_unique("case id", [c.id for c in self.cases])
        known = set(provider_ids)
        if self.judge is not None and self.judge not in known:
            raise ValueError(
                f"judge {self.judge!r} is not a provider id ({', '.join(provider_ids)})"
            )
        for case in self.cases:
            for a in case.assertions:
                if isinstance(a, LLMJudgeAssertion):
                    judge = a.provider or self.judge
                    if judge is None:
                        raise ValueError(
                            f"case {case.id!r}: llm_judge needs 'provider' or a suite-level 'judge'"
                        )
                    if judge not in known:
                        raise ValueError(f"case {case.id!r}: judge provider {judge!r} not defined")
        return self

    def provider(self, provider_id: str) -> OpenAIProviderSpec | MockProviderSpec:
        for p in self.providers:
            if p.id == provider_id:
                return p
        raise KeyError(provider_id)


def _require_unique(label: str, values: list[str]) -> None:
    seen: set[str] = set()
    for v in values:
        if v in seen:
            raise ValueError(f"duplicate {label}: {v!r}")
        seen.add(v)


# ------------------------------------------------------------------------ loading


def _resolve_file_ref(value: Any, base: Path, *, parse: bool) -> Any:
    if not (isinstance(value, str) and value.startswith(FILE_REF)):
        return value
    path = (base / value[len(FILE_REF) :]).resolve()
    if not path.is_file():
        raise ConfigError(f"referenced file not found: {path}")
    text = path.read_text(encoding="utf-8")
    if not parse:
        return text
    if path.suffix == ".json":
        return json.loads(text)
    return yaml.safe_load(text)


def load_suite(path: str | Path) -> Suite:
    """Load and validate a suite file. `file://` values for `system` and `tools`
    are resolved relative to the suite file."""
    suite_path = Path(path)
    if not suite_path.is_file():
        raise ConfigError(f"suite file not found: {suite_path}")
    try:
        raw = yaml.safe_load(suite_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{suite_path}: invalid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"{suite_path}: expected a mapping at the top level")

    base = suite_path.parent
    if "system" in raw:
        raw["system"] = _resolve_file_ref(raw["system"], base, parse=False)
    if "tools" in raw:
        raw["tools"] = _resolve_file_ref(raw["tools"], base, parse=True)

    try:
        return Suite.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"{suite_path}: invalid suite\n{_format_errors(exc)}") from exc


def _format_errors(exc: ValidationError) -> str:
    lines = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err["loc"])
        lines.append(f"  - {loc}: {err['msg']}")
    return "\n".join(lines)
