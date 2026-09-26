from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from harness.config import Check
from harness.contracts import HarnessError
from harness.runner import CheckResult, run_check


def execute(
    tmp_path: Path,
    code: str,
    *,
    kind: str = "command",
    timeout: float = 2,
    limit: int = 4096,
    allowed: bool = True,
    pattern: str | None = None,
) -> CheckResult:
    repo, run = tmp_path / "repo", tmp_path / "run"
    repo.mkdir(exist_ok=True)
    run.mkdir(exist_ok=True)
    return run_check(
        Check("test", (sys.executable, "-c", code), kind, pattern),
        repo,
        run,
        allowed=allowed,
        timeout=timeout,
        output_limit=limit,
        sequence=1,
        fingerprint="fixture",
        phase="test",
    )


def test_no_execution_without_opt_in(tmp_path: Path) -> None:
    with pytest.raises(HarnessError, match="disabled"):
        execute(tmp_path, "raise SystemExit(0)", allowed=False)


def test_actual_exit_code_not_output_claim(tmp_path: Path) -> None:
    result = execute(tmp_path, "print('Everything passed'); raise SystemExit(1)")
    assert not result.passed
    assert result.exit_code == 1
    assert result.reason == "nonzero_exit"


@pytest.mark.parametrize(
    "output,passed,count",
    [
        ("Ran 0 tests in 0s\\nOK", False, 0),
        ("Ran 1 test in 0s\\nOK (skipped=1)", False, 0),
        ("Ran 2 tests in 0s\\nOK (skipped=1)", True, 1),
        ("Ran 7 tests in 0s\\nOK", True, 7),
        ("OK", False, 0),
    ],
)
def test_unittest_requires_executed_tests(
    tmp_path: Path,
    output: str,
    passed: bool,
    count: int,
) -> None:
    result = execute(tmp_path, f"print('{output}')", kind="unittest")
    assert result.passed is passed
    assert result.tests_executed == count


@pytest.mark.parametrize(
    "output,passed",
    [
        ("2 passed in 0.1s", True),
        ("2 skipped in 0.1s", False),
        ("no tests ran", False),
    ],
)
def test_pytest_requires_passed_tests(tmp_path: Path, output: str, passed: bool) -> None:
    result = execute(tmp_path, f"print({output!r})", kind="pytest")
    assert result.passed is passed


def test_timeout_is_not_success(tmp_path: Path) -> None:
    result = execute(tmp_path, "import time; time.sleep(10)", timeout=0.15)
    assert result.reason == "timeout"
    assert not result.passed
    assert result.duration_seconds < 2


def test_infinite_output_is_bounded(tmp_path: Path) -> None:
    result = execute(tmp_path, "while True: print('x' * 1000)", limit=2000)
    assert result.reason == "output_limit"
    assert not result.passed
    assert (tmp_path / "run" / result.log).stat().st_size == 2000


def test_credentials_not_inherited(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-value-not-a-real-key")
    monkeypatch.setenv("AI_API_KEY", "runtime-credential-test-placeholder")
    monkeypatch.setenv("CUSTOM_SECRET", "another-test-value")
    result = execute(
        tmp_path,
        (
            "import os; assert 'OPENAI_API_KEY' not in os.environ; "
            "assert 'AI_API_KEY' not in os.environ; "
            "assert 'CUSTOM_SECRET' not in os.environ; print(os.environ['HOME'])"
        ),
    )
    assert result.passed
    assert str(tmp_path / "run/command-home") in result.output


def test_required_output_is_enforced(tmp_path: Path) -> None:
    result = execute(tmp_path, "print('wrong')", pattern="^expected$")
    assert result.reason == "required_output_missing"
    assert not result.passed


def test_invalid_utf8_output_is_retained_with_replacement(tmp_path: Path) -> None:
    result = execute(tmp_path, "import os; os.write(1, b'\\xff')")
    assert result.passed
    assert "\ufffd" in result.output


def test_child_holding_output_pipe_is_terminated(tmp_path: Path) -> None:
    result = execute(
        tmp_path,
        (
            "import subprocess, sys; "
            "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(20)']); "
            "print(p.pid, flush=True)"
        ),
        timeout=0.2,
    )
    assert result.reason == "timeout"
    pid = int(result.output.strip())
    # A killed orphan can briefly remain a zombie before the OS reaps it.
    import subprocess

    probe = subprocess.run(
        ["ps", "-o", "stat=", "-p", str(pid)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert probe.returncode != 0 or probe.stdout.strip().startswith("Z")
    assert os.getpid() != pid
