from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from harness.contracts import HarnessError

DEFAULT_PROTECTED = (
    "tests/*",
    "*/tests/*",
    "test_*.py",
    "*/test_*.py",
    "*_test.py",
    "*_test.go",
    "*.test.*",
    "*.spec.*",
    "conftest.py",
    "*/conftest.py",
    "pytest.ini",
    "tox.ini",
    ".github/*",
)


@dataclass(frozen=True)
class Limits:
    max_steps: int = 24
    max_total_tokens: int = 50_000
    max_output_tokens: int = 4096
    max_seconds: int = 180
    command_timeout: int = 30
    max_output_bytes: int = 131_072
    max_context_chars: int = 24_000
    max_file_bytes: int = 1_048_576
    max_repo_bytes: int = 16_777_216
    max_files: int = 2_000

    def __post_init__(self) -> None:
        for item in fields(self):
            value = getattr(self, item.name)
            if type(value) is not int or value <= 0:
                raise HarnessError(f"limits.{item.name} must be a positive integer")
        if self.max_context_chars < 4_096:
            raise HarnessError("limits.max_context_chars must be at least 4096")


@dataclass(frozen=True)
class Check:
    name: str
    argv: tuple[str, ...]
    kind: str = "command"
    success_pattern: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not re.fullmatch(r"[A-Za-z][\w-]{0,39}", self.name):
            raise HarnessError("Check names must be 1-40 letters, digits, underscores or hyphens")
        if not self.argv or any(
            not isinstance(arg, str) or not arg or "\x00" in arg for arg in self.argv
        ):
            raise HarnessError(f"Check {self.name} needs a nonempty string argv array")
        if self.kind not in ("command", "unittest", "pytest"):
            raise HarnessError(f"Unsupported check kind: {self.kind}")
        if self.success_pattern is not None:
            if not isinstance(self.success_pattern, str) or not self.success_pattern:
                raise HarnessError("success_pattern must be a nonempty regular expression")
            try:
                re.compile(self.success_pattern)
            except re.error as exc:
                raise HarnessError(f"Invalid success_pattern for check {self.name}") from exc


@dataclass(frozen=True)
class Config:
    checks: tuple[Check, ...] = ()
    protected: tuple[str, ...] = DEFAULT_PROTECTED
    limits: Limits = field(default_factory=Limits)

    def __post_init__(self) -> None:
        if len({check.name for check in self.checks}) != len(self.checks):
            raise HarnessError("Check names must be unique")
        if any(not isinstance(pattern, str) or not pattern for pattern in self.protected):
            raise HarnessError("protected entries must be nonempty glob patterns")


def _object(value: Any, allowed: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise HarnessError(f"{label} must be an object")
    unknown = set(value) - allowed
    if unknown:
        raise HarnessError(f"Unknown {label} fields: {', '.join(sorted(unknown))}")
    return dict(value)


def _unique_config_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, value in pairs:
        if name in result:
            raise HarnessError(f"Duplicate configuration key: {name}")
        result[name] = value
    return result


def load_config(path: Path | None) -> Config:
    if path is None:
        return Config()
    try:
        with path.open(encoding="utf-8") as stream:
            text = stream.read(65_537)
        if len(text) > 65_536:
            raise HarnessError("Configuration exceeds 65536 characters")
        raw = json.loads(text, object_pairs_hook=_unique_config_keys)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise HarnessError(f"Cannot read valid configuration JSON from {path}") from exc
    obj = _object(raw, {"checks", "protected", "limits"}, "config")
    limit_values = _object(obj.get("limits", {}), {item.name for item in fields(Limits)}, "limits")
    raw_checks = obj.get("checks", [])
    if not isinstance(raw_checks, list):
        raise HarnessError("checks must be an array")
    checks = []
    for raw_check in raw_checks:
        item = _object(raw_check, {"name", "argv", "kind", "success_pattern"}, "check")
        if "name" not in item or "argv" not in item or not isinstance(item["argv"], list):
            raise HarnessError("Each check needs name and argv fields")
        checks.append(
            Check(
                item["name"],
                tuple(item["argv"]),
                item.get("kind", "command"),
                item.get("success_pattern"),
            )
        )
    extra_protected = obj.get("protected", [])
    if not isinstance(extra_protected, list):
        raise HarnessError("protected must be an array")
    return Config(tuple(checks), DEFAULT_PROTECTED + tuple(extra_protected), Limits(**limit_values))
