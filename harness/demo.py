from __future__ import annotations

import json
from importlib.resources import files
from pathlib import Path
from typing import Any

from harness.config import Check, Config
from harness.contracts import Action, CallLimits, ModelError, ModelReply
from harness.workspace import digest

DEMO_TASK = (
    "Fix contains(value, lower, upper) so both endpoints of a closed interval are included. "
    "Preserve rejection of reversed intervals and behavior for values outside the interval."
)
DEMO_CONFIG = Config(
    checks=(
        Check("tests", ("{python}", "-m", "unittest", "discover", "-s", "tests", "-v"), "unittest"),
    )
)


def create_demo_repository(destination: Path) -> None:
    if destination.exists():
        raise ModelError("Demo destination must not already exist")
    destination.mkdir(parents=True)
    (destination / "tests").mkdir()
    templates = files("harness").joinpath("fixtures")
    (destination / "intervals.py").write_bytes(templates.joinpath("intervals.py.txt").read_bytes())
    (destination / "tests" / "test_intervals.py").write_bytes(
        templates.joinpath("test_intervals.py.txt").read_bytes()
    )


class DemoModel:
    """Scripted fixture actions; never represents real model performance."""

    def __init__(self) -> None:
        self.index = 0

    def choose_action(
        self,
        instructions: str,
        context: str,
        tools: list[dict[str, Any]],
        limits: CallLimits,
    ) -> ModelReply:
        state = json.loads(context)
        template = files("harness").joinpath("fixtures").joinpath("intervals.py.txt").read_bytes()
        actions = [
            Action("list_files", {"pattern": "*", "offset": 0}),
            Action("read_file", {"path": "intervals.py", "start_line": 1, "end_line": 20}),
            Action(
                "record_plan",
                {
                    "plan": "Make both ends inclusive; preserve validation; run tests.",
                },
            ),
            Action(
                "edit_file",
                {
                    "path": "intervals.py",
                    "expected_sha256": digest(template),
                    "old_text": "return lower < value < upper",
                    "new_text": "return lower <= value <= upper",
                },
            ),
            Action("run_check", {"name": "tests"}),
            Action(
                "finish",
                {
                    "summary": "Scripted demo: fixed endpoints; final checks run independently.",
                },
            ),
        ]
        if self.index >= len(actions) or "task" not in state:
            raise ModelError("Demo action sequence exhausted unexpectedly")
        action = actions[self.index]
        self.index += 1
        return ModelReply(action, 0, 0)
