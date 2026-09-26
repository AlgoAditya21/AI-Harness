from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from harness.config import Check, Config, Limits
from harness.contracts import Action, CallLimits, HarnessError, ModelError, ModelReply
from harness.demo import DEMO_CONFIG, DEMO_TASK, DemoModel, create_demo_repository
from harness.engine import Harness
from harness.workspace import digest


class SequenceModel:
    def __init__(self, actions: list[Any]) -> None:
        self.actions = iter(actions)
        self.contexts: list[dict[str, Any]] = []

    def choose_action(
        self,
        instructions: str,
        context: str,
        tools: list[dict[str, Any]],
        limits: CallLimits,
    ) -> ModelReply:
        self.contexts.append(json.loads(context))
        action = next(self.actions, ModelError("Script exhausted"))
        if isinstance(action, BaseException):
            raise action
        return ModelReply(action, 10, 5)


def make_run(
    tmp_path: Path,
    model: Any,
    config: Config | None = None,
    task: str = DEMO_TASK,
) -> Harness:
    source = tmp_path / "repo"
    create_demo_repository(source)
    return Harness(
        model,
        source,
        task,
        tmp_path / "runs",
        config or Config(),
        allow_host_execution=True,
    )


def test_full_demo_fixes_bug_and_replays_in_fresh_repository(tmp_path: Path) -> None:
    harness = make_run(tmp_path, DemoModel(), DEMO_CONFIG)
    original = (harness.source / "intervals.py").read_bytes()
    result = harness.run()
    assert result["status"] == "verified"
    assert result["steps"] == 6
    assert result["reported_tokens"] == 0
    assert [check["passed"] for check in result["checks"]] == [False, True, True]
    assert result["checks"][-1]["phase"] == "final"
    assert result["checks"][-1]["tests_executed"] == 7
    assert (harness.source / "intervals.py").read_bytes() == original
    assert len(result["changes"]) == 1
    assert result["changes"][0]["path"] == "intervals.py"
    assert json.loads((harness.run_dir / "result.json").read_text()) == result

    replay = tmp_path / "fresh-replay"
    shutil.copytree(harness.source, replay)
    for change in result["changes"]:
        destination = replay / change["path"]
        assert digest(destination.read_bytes()) == change["before_sha256"]
        destination.write_bytes((harness.run_dir / "changed-files" / change["path"]).read_bytes())
        assert digest(destination.read_bytes()) == change["after_sha256"]
    validation = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
        cwd=replay,
        capture_output=True,
        text=True,
        check=False,
    )
    assert validation.returncode == 0, validation.stderr
    assert "Ran 7 tests" in validation.stderr
    oracle = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from intervals import contains\n"
                "for lo in range(-3, 4):\n"
                " for hi in range(lo, 4):\n"
                "  for x in range(-5, 6):\n"
                "   assert contains(x, lo, hi) == (lo <= x <= hi)\n"
            ),
        ],
        cwd=replay,
        capture_output=True,
        text=True,
        check=False,
    )
    assert oracle.returncode == 0, oracle.stderr


def test_finish_without_checks_is_unverified(tmp_path: Path) -> None:
    model = SequenceModel([Action("finish", {"summary": "No tests are configured"})])
    result = make_run(tmp_path, model).run()
    assert result["status"] == "unverified"


def test_claim_of_success_does_not_override_failing_tests(tmp_path: Path) -> None:
    model = SequenceModel([Action("finish", {"summary": "All tests pass!"})])
    result = make_run(tmp_path, model, DEMO_CONFIG).run()
    assert result["status"] == "failed"
    assert all(not check["passed"] for check in result["checks"])
    assert any("finish_rejected" in str(item) for item in model.contexts)


def test_successful_check_invalidated_after_another_edit(tmp_path: Path) -> None:
    source = tmp_path / "repo"
    create_demo_repository(source)
    old = (source / "intervals.py").read_text()
    fixed = old.replace("lower < value < upper", "lower <= value <= upper")
    model = SequenceModel(
        [
            Action("record_plan", {"plan": "Fix and verify endpoints"}),
            Action(
                "edit_file",
                {
                    "path": "intervals.py",
                    "expected_sha256": digest(old.encode()),
                    "old_text": "lower < value < upper",
                    "new_text": "lower <= value <= upper",
                },
            ),
            Action("run_check", {"name": "tests"}),
            Action(
                "edit_file",
                {
                    "path": "intervals.py",
                    "expected_sha256": digest(fixed.encode()),
                    "old_text": "lower <= value <= upper",
                    "new_text": "False",
                },
            ),
            Action("finish", {"summary": "Tests passed earlier"}),
        ]
    )
    result = Harness(
        model,
        source,
        DEMO_TASK,
        tmp_path / "runs",
        DEMO_CONFIG,
        allow_host_execution=True,
    ).run()
    assert result["status"] == "failed"
    assert result["checks"][1]["passed"]
    assert not result["checks"][-1]["passed"]
    assert result["checks"][1]["fingerprint"] != result["checks"][-1]["fingerprint"]


def test_invalid_tool_calls_are_recoverable_and_logged(tmp_path: Path) -> None:
    model = SequenceModel(
        [
            Action("read_file", {"path": "../outside", "start_line": 1, "end_line": 2}),
            Action("not_a_tool", {}),
            Action("finish", {"summary": "Stopped without changing code"}),
        ]
    )
    harness = make_run(tmp_path, model)
    result = harness.run()
    assert result["status"] == "unverified"
    events = [
        json.loads(line) for line in (harness.run_dir / "events.jsonl").read_text().splitlines()
    ]
    assert len([event for event in events if event["type"] == "tool_error"]) == 2


def test_repeated_actions_stop(tmp_path: Path) -> None:
    action = Action("read_file", {"path": "missing", "start_line": 1, "end_line": 2})
    result = make_run(tmp_path, SequenceModel([action] * 10)).run()
    assert result["status"] == "budget_exhausted"
    assert result["steps"] == 3


def test_step_budget_is_enforced(tmp_path: Path) -> None:
    config = Config(limits=Limits(max_steps=1))
    model = SequenceModel([Action("list_files", {"pattern": "*", "offset": 0})])
    result = make_run(tmp_path, model, config).run()
    assert result["status"] == "budget_exhausted"
    assert result["steps"] == 1


def test_small_token_budget_prevents_any_request(tmp_path: Path) -> None:
    model = SequenceModel([])
    config = Config(limits=Limits(max_total_tokens=10))
    result = make_run(tmp_path, model, config).run()
    assert result["status"] == "budget_exhausted"
    assert model.contexts == []


def test_model_output_limit_is_configurable(tmp_path: Path) -> None:
    harness = make_run(
        tmp_path,
        SequenceModel([]),
        Config(limits=Limits(max_output_tokens=1234)),
    )
    import time

    harness.started = time.monotonic()
    limits, reservation = harness._call_limits("small context")
    assert limits.max_output_tokens == 1234
    assert reservation > limits.max_output_tokens


def test_retry_is_bounded_and_unknown_usage_is_accounted(tmp_path: Path) -> None:
    model = SequenceModel(
        [
            ModelError("Temporary service error", retryable=True),
            Action("finish", {"summary": "Recovered"}),
        ]
    )
    harness = make_run(tmp_path, model)
    result = harness.run()
    assert result["status"] == "unverified"
    assert result["steps"] == 2
    assert result["usage_estimated"]
    assert result["accounted_tokens"] > result["reported_tokens"] == 15


def test_permanent_model_error_is_not_retried(tmp_path: Path) -> None:
    model = SequenceModel([ModelError("Unauthorized")])
    result = make_run(tmp_path, model).run()
    assert result["status"] == "failed"
    assert len(model.contexts) == 1


def test_cancellation_is_persisted(tmp_path: Path) -> None:
    harness = make_run(tmp_path, SequenceModel([KeyboardInterrupt()]))
    result = harness.run()
    assert result["status"] == "cancelled"
    assert json.loads((harness.run_dir / "checkpoint.json").read_text())["status"] == "cancelled"


def test_no_checks_run_without_host_opt_in(tmp_path: Path) -> None:
    source = tmp_path / "repo"
    create_demo_repository(source)
    with pytest.raises(HarnessError, match="allow-host-execution"):
        Harness(DemoModel(), source, DEMO_TASK, tmp_path / "runs", DEMO_CONFIG)


def test_check_changing_source_cannot_be_verification(tmp_path: Path) -> None:
    check = Check(
        "mutator",
        (
            sys.executable,
            "-c",
            "from pathlib import Path; Path('intervals.py').write_text('')",
        ),
    )
    model = SequenceModel([])
    result = make_run(tmp_path, model, Config(checks=(check,))).run()
    assert result["status"] == "failed"
    assert "changed repository" in result["summary"]
    assert result["checks"][0]["passed"] is False
    assert result["checks"][0]["reason"] == "workspace_changed"
    assert not model.contexts


def test_all_required_checks_must_pass(tmp_path: Path) -> None:
    checks = (
        Check("pass", (sys.executable, "-c", "print('ok')")),
        Check("fail", (sys.executable, "-c", "raise SystemExit(1)")),
    )
    model = SequenceModel([Action("finish", {"summary": "Done"})])
    result = make_run(tmp_path, model, Config(checks=checks)).run()
    assert result["status"] == "failed"
    assert [item["passed"] for item in result["checks"]] == [True, False, True, False]


def test_plan_and_task_survive_history_trimming(tmp_path: Path) -> None:
    actions = [
        Action("record_plan", {"plan": "RETAIN_THIS_PLAN"}),
        *[Action("list_files", {"pattern": f"*{i}*", "offset": 0}) for i in range(10)],
        Action("finish", {"summary": "Done"}),
    ]
    model = SequenceModel(actions)
    config = replace(Config(), limits=Limits(max_context_chars=4096))
    result = make_run(tmp_path, model, config).run()
    assert result["status"] == "unverified"
    assert model.contexts[-1]["plan"] == "RETAIN_THIS_PLAN"
    assert model.contexts[-1]["task"] == DEMO_TASK
    assert model.contexts[-1]["omitted_observation_count"] > 0
    assert all(len(json.dumps(context)) <= 4096 for context in model.contexts)


def test_long_check_log_is_retrievable_by_character_offset(tmp_path: Path) -> None:
    config = Config(
        checks=(
            Check(
                "long",
                (sys.executable, "-c", "print('A' * 15000 + 'TAIL')"),
            ),
        )
    )
    model = SequenceModel(
        [
            Action(
                "read_chunk",
                {
                    "source": "check_log",
                    "path": "logs/0001-baseline-long.log",
                    "offset": 15000,
                    "length": 5,
                },
            ),
            Action("finish", {"summary": "Inspected log tail"}),
        ]
    )
    harness = make_run(tmp_path, model, config)
    result = harness.run()
    assert result["status"] == "verified"
    observations = model.contexts[-1]["recent_observations"]
    assert any(event.get("result", {}).get("content") == "TAIL\n" for event in observations)


def test_private_artifacts_cannot_be_read_as_logs(tmp_path: Path) -> None:
    model = SequenceModel(
        [
            Action("read_check_log", {"log": "request.json", "start_line": 1, "end_line": 2}),
            Action("finish", {"summary": "Stopped"}),
        ]
    )
    result = make_run(tmp_path, model).run()
    assert result["status"] == "unverified"
    assert any("Only logs" in str(event) for event in model.contexts[-1]["recent_observations"])


def test_context_overflow_fails_without_dropping_task(tmp_path: Path) -> None:
    model = SequenceModel([])
    result = make_run(
        tmp_path,
        model,
        Config(limits=Limits(max_context_chars=4096)),
        task="x" * 5000,
    ).run()
    assert result["status"] == "budget_exhausted"
    assert not model.contexts


def test_transient_errors_stop_after_three_requests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("harness.engine.time.sleep", lambda seconds: None)
    model = SequenceModel([ModelError("Temporary", retryable=True)] * 5)
    config = Config(limits=Limits(max_total_tokens=100_000))
    result = make_run(tmp_path, model, config).run()
    assert result["status"] == "failed"
    assert result["model_calls"] == 3


def test_checkpoint_failure_is_an_explicit_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_write(path: Path, value: object) -> None:
        raise OSError("simulated disk failure")

    harness = make_run(tmp_path, SequenceModel([]))
    monkeypatch.setattr("harness.engine.write_json", fail_write)
    with pytest.raises(OSError, match="simulated disk"):
        harness.run()
    assert not (harness.run_dir / "result.json").exists()
