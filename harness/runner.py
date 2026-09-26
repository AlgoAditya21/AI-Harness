from __future__ import annotations

import os
import re
import selectors
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from harness.config import Check
from harness.contracts import HarnessError


@dataclass(frozen=True)
class CheckResult:
    name: str
    argv: tuple[str, ...]
    kind: str
    passed: bool
    exit_code: int | None
    reason: str
    tests_executed: int | None
    duration_seconds: float
    output: str
    log: str
    fingerprint: str
    phase: str

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "argv": list(self.argv)}


def _stop_group(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass  # The owned process group has already exited.
    process.wait()


def _test_count(check: Check, output: str) -> int | None:
    if check.kind == "unittest":
        counts = re.findall(r"^Ran (\d+) tests?\b", output, re.MULTILINE)
        if not counts:
            return 0
        skipped = re.findall(r"^OK \(skipped=(\d+)\)", output, re.MULTILINE)
        return max(0, int(counts[-1]) - (int(skipped[-1]) if skipped else 0))
    if check.kind == "pytest":
        counts = re.findall(r"\b(\d+) passed\b", output)
        return int(counts[-1]) if counts else 0
    return None


def run_check(
    check: Check,
    workspace: Path,
    run_dir: Path,
    *,
    allowed: bool,
    timeout: float,
    output_limit: int,
    sequence: int,
    fingerprint: str,
    phase: str,
) -> CheckResult:
    if not allowed:
        raise HarnessError(
            "Host execution is disabled; use --allow-host-execution for trusted code"
        )
    if os.name != "posix":
        raise HarnessError("Host execution currently supports macOS and Linux only")
    if timeout <= 0:
        raise HarnessError("No time remains for this check")
    logs = run_dir / "logs"
    logs.mkdir(exist_ok=True)
    home, temporary = run_dir / "command-home", run_dir / "command-tmp"
    home.mkdir(exist_ok=True)
    temporary.mkdir(exist_ok=True)
    env = {
        name: os.environ[name] for name in ("PATH", "LANG", "LC_ALL", "TZ") if name in os.environ
    }
    env.update(
        {
            "HOME": str(home),
            "TMPDIR": str(temporary),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
            "NO_COLOR": "1",
            "TERM": "dumb",
        }
    )
    argv = tuple(sys.executable if arg == "{python}" else arg for arg in check.argv)
    started = time.monotonic()
    log_path = logs / f"{sequence:04d}-{phase}-{check.name}.log"
    reason = "completed"
    output = bytearray()
    try:
        process = subprocess.Popen(
            argv,
            cwd=workspace,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    except OSError as exc:
        raise HarnessError(
            f"Could not start check {check.name}; verify executable and environment"
        ) from exc
    assert process.stdout is not None
    try:
        with selectors.DefaultSelector() as selector, log_path.open("xb") as log:
            os.chmod(log_path, 0o600)
            selector.register(process.stdout, selectors.EVENT_READ)
            while selector.get_map():
                if time.monotonic() - started >= timeout:
                    reason = "timeout"
                    break
                for key, _ in selector.select(timeout=min(0.05, timeout)):
                    chunk = os.read(key.fd, 8192)
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    remaining = output_limit - len(output)
                    kept = chunk[:remaining]
                    output.extend(kept)
                    log.write(kept)
                    if len(chunk) > remaining:
                        reason = "output_limit"
                        break
                if reason != "completed":
                    break
            if reason == "completed":
                remaining_time = timeout - (time.monotonic() - started)
                try:
                    process.wait(timeout=max(0.001, remaining_time))
                except subprocess.TimeoutExpired:
                    reason = "timeout"
            log.flush()
            os.fsync(log.fileno())
    finally:
        _stop_group(process)
        process.stdout.close()
    text = output.decode("utf-8", errors="replace")
    executed = _test_count(check, text)
    passed = reason == "completed" and process.returncode == 0
    if reason == "completed" and process.returncode != 0:
        reason = "nonzero_exit"
    if passed and executed == 0:
        passed, reason = False, "no_tests_executed"
    if passed and check.success_pattern is not None and not re.search(check.success_pattern, text):
        passed, reason = False, "required_output_missing"
    excerpt = text
    if len(excerpt) > 6000:
        excerpt = (
            excerpt[:2000] + "\n[OUTPUT EXCERPT; full bounded log on disk]\n" + excerpt[-4000:]
        )
    return CheckResult(
        check.name,
        argv,
        check.kind,
        passed,
        process.returncode,
        reason,
        executed,
        round(time.monotonic() - started, 3),
        excerpt,
        str(log_path.relative_to(run_dir)),
        fingerprint,
        phase,
    )
