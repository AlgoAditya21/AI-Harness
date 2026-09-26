from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class HarnessError(Exception):
    """An expected configuration, tool, or execution failure."""


class ModelError(HarnessError):
    """A model request could not produce a valid action."""

    def __init__(self, message: str, *, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable


class ModelActionError(ModelError):
    """A billed, completed response needs a corrected action, not a transport retry."""

    def __init__(self, message: str, input_tokens: int, output_tokens: int):
        super().__init__(message)
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class BudgetError(HarnessError):
    """A run or model-call budget has been exhausted."""


@dataclass(frozen=True)
class Action:
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ModelReply:
    action: Action
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class CallLimits:
    max_output_tokens: int
    timeout_seconds: float


@dataclass(frozen=True)
class ModelConfiguration:
    provider: str
    protocol: str
    base_url: str
    thinking: str
    token_limit_parameter: str


class Model(Protocol):
    def choose_action(
        self,
        instructions: str,
        context: str,
        tools: list[dict[str, Any]],
        limits: CallLimits,
    ) -> ModelReply:
        """Return one action; tools use named function JSON schemas."""
        ...
