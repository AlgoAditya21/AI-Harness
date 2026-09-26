from __future__ import annotations

import json
import math
import os
from typing import Any, NoReturn

from openai import APIConnectionError, APIError, APIStatusError

from harness.contracts import CallLimits, ModelError


def runtime_api_key() -> str:
    key = os.environ.get("AI_API_KEY")
    if not key or not key.strip():
        raise ModelError("AI_API_KEY must be set in the environment before a live run.")
    if any(
        character.isspace() or ord(character) < 32 or ord(character) == 127 for character in key
    ):
        raise ModelError(
            "AI_API_KEY contains whitespace or control characters; re-enter it privately."
        )
    return key


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON object key.")
        result[key] = value
    return result


def _reject_constant(value: str) -> NoReturn:
    raise ValueError("Non-finite JSON number.")


def _finite_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("Non-finite JSON number.")
    return number


def decode_json(value: str | bytes, message: str) -> Any:
    try:
        return json.loads(
            value,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
            parse_float=_finite_float,
        )
    except (ValueError, RecursionError) as exc:
        raise ModelError(message) from exc


def validate_limits(limits: CallLimits) -> float:
    if type(limits.max_output_tokens) is not int or limits.max_output_tokens <= 0:
        raise ModelError("The output-token limit must be a positive integer.")
    if type(limits.timeout_seconds) not in (int, float):
        raise ModelError("The request timeout must be a positive finite number.")
    try:
        timeout = float(limits.timeout_seconds)
    except OverflowError as exc:
        raise ModelError("The request timeout must be a positive finite number.") from exc
    if not math.isfinite(timeout) or timeout <= 0:
        raise ModelError("The request timeout must be a positive finite number.")
    return timeout


def tool_names(tools: list[dict[str, Any]]) -> set[str]:
    names: set[str] = set()
    for tool in tools:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            raise ModelError("Only function tools are supported.")
        name = tool.get("name")
        if not isinstance(name, str) or not name.strip() or name in names:
            raise ModelError("Function tools must have unique, nonempty names.")
        names.add(name)
    if not names:
        raise ModelError("At least one function tool is required.")
    return names


def parse_usage(
    usage: Any,
    input_field: str,
    output_field: str,
    provider: str,
) -> tuple[int, int]:
    if not isinstance(usage, dict):
        raise ModelError(f"{provider} did not report valid token usage.")
    input_tokens, output_tokens = usage.get(input_field), usage.get(output_field)
    if (
        type(input_tokens) is not int
        or input_tokens < 0
        or type(output_tokens) is not int
        or output_tokens < 0
    ):
        raise ModelError(f"{provider} did not report valid token usage.")
    return input_tokens, output_tokens


def transient_api_error(error: APIError) -> bool:
    return isinstance(error, APIConnectionError) or (
        isinstance(error, APIStatusError)
        and (error.status_code == 429 or 500 <= error.status_code < 600)
    )
