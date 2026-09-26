import json
from pathlib import Path

import pytest

from harness.cli import main
from harness.config import Check, Config, Limits, load_config
from harness.contracts import Action, HarnessError
from harness.tools import Tools, validate_action
from harness.workspace import Workspace


@pytest.mark.parametrize(
    "raw",
    [
        {"unexpected": True},
        {"checks": {}},
        {"checks": [{"name": "x"}]},
        {"checks": [{"name": "x", "argv": "python -m pytest"}]},
        {"checks": [{"name": "x", "argv": ["python"], "kind": "unknown"}]},
        {"checks": [{"name": "../bad", "argv": ["python"]}]},
        {"limits": {"max_steps": True}},
        {"limits": {"max_steps": -1}},
        {"limits": {"max_steps": 1.5}},
        {"limits": {"made_up": 1}},
        {"protected": "tests/*"},
        {"protected": [False]},
    ],
)
def test_invalid_config_is_explicit(tmp_path: Path, raw: object) -> None:
    path = tmp_path / "config.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(HarnessError):
        load_config(path)


def test_duplicate_check_names_rejected(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"checks": [{"name": "x", "argv": ["true"]}] * 2}))
    with pytest.raises(HarnessError, match="unique"):
        load_config(path)


def test_invalid_pattern_rejected() -> None:
    with pytest.raises(HarnessError, match="success_pattern"):
        Check("test", ("true",), success_pattern="[")


def test_context_minimum() -> None:
    with pytest.raises(HarnessError, match="4096"):
        Limits(max_context_chars=100)


@pytest.mark.parametrize(
    "action",
    [
        Action("read_file", {"path": "a", "start_line": True, "end_line": 2}),
        Action("read_file", {"path": "a", "start_line": 0, "end_line": 2}),
        Action("finish", {"summary": "x", "extra": "y"}),
        Action("finish", {"summary": "x" * 4001}),
        Action("finish", {}),
        Action("run_any_shell", {"command": "arbitrary"}),
    ],
)
def test_invalid_action_schema(action: Action) -> None:
    with pytest.raises(HarnessError):
        validate_action(action)


def test_live_mode_requires_upload_permission(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(
        [
            "run",
            "--repo",
            ".",
            "--task",
            "test",
            "--model",
            "configured-model",
            "--provider",
            "deepseek",
            "--base-url",
            "https://evaluator.invalid/v1",
        ]
    )
    assert code == 2
    assert "allow-model-upload" in capsys.readouterr().err


def test_key_is_not_required_for_demo(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["demo", "--output", str(tmp_path / "runs"), "--allow-host-execution"])
    assert code == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "verified"
    assert result["model"] == "scripted-demo-not-an-llm"


def test_demo_requires_explicit_execution(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["demo", "--output", str(tmp_path / "runs")])
    assert code == 2
    assert "allow-host-execution" in capsys.readouterr().err


def test_init_demo_does_not_overwrite(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    destination = tmp_path / "repo"
    assert main(["init-demo", "--destination", str(destination)]) == 0
    capsys.readouterr()
    assert main(["init-demo", "--destination", str(destination)]) == 2
    assert "already exist" in capsys.readouterr().err


@pytest.mark.parametrize(
    "text",
    [
        '{"limits": {"max_steps": 1, "max_steps": 100}}',
        '{"checks": [], "checks": []}',
    ],
)
def test_duplicate_json_keys_rejected(tmp_path: Path, text: str) -> None:
    path = tmp_path / "config.json"
    path.write_text(text)
    with pytest.raises(HarnessError, match="Duplicate"):
        load_config(path)


def test_invalid_unicode_tool_argument_rejected() -> None:
    with pytest.raises(HarnessError, match="Unicode"):
        validate_action(Action("finish", {"summary": "\ud800"}))


def test_large_search_is_paginated_and_long_lines_are_readable(tmp_path: Path) -> None:
    source, run = tmp_path / "repo", tmp_path / "run"
    source.mkdir()
    run.mkdir()
    (source / "many.py").write_text("match\n" * 10_000)
    (source / "long.py").write_text("a" * 15000 + "TAIL")
    workspace = Workspace(source, run, Config())
    tools = Tools(workspace, lambda name: {}, lambda plan: None, lambda *args: {})
    result = tools.execute(Action("search", {"query": "match", "pattern": "*", "offset": 9990}))
    assert result["total"] == 10_000
    assert len(result["matches"]) == 10
    assert result["matches"][0]["line"] == 9991
    assert result["next_offset"] is None
    result = tools.execute(
        Action(
            "read_chunk",
            {
                "source": "file",
                "path": "long.py",
                "offset": 15000,
                "length": 4,
            },
        )
    )
    assert result["content"] == "TAIL"
    assert result["next_offset"] is None
