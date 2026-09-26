from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from harness.config import Config
from harness.contracts import (
    BudgetError,
    CallLimits,
    HarnessError,
    Model,
    ModelActionError,
    ModelConfiguration,
    ModelError,
)
from harness.prompt import SYSTEM_PROMPT
from harness.runner import CheckResult, run_check
from harness.tools import TOOLS, Tools, validate_action
from harness.workspace import Workspace, digest, write_json


class Harness:
    def __init__(
        self,
        model: Model,
        source: Path,
        task: str,
        output: Path,
        config: Config,
        *,
        allow_host_execution: bool = False,
        model_label: str = "scripted",
        model_configuration: ModelConfiguration | None = None,
    ):
        if not task.strip():
            raise HarnessError("Task must not be empty")
        if len(task) > 8000:
            raise HarnessError("Task exceeds 8000 characters; provide a focused task")
        if config.checks and not allow_host_execution:
            raise HarnessError(
                "Configured checks require --allow-host-execution; use only with trusted code"
            )
        if output.resolve().is_relative_to(source.resolve()):
            raise HarnessError("Output directory must be outside the source repository")
        self.model, self.source, self.task, self.config = model, source, task, config
        self.allow_host_execution, self.model_label = allow_host_execution, model_label
        self.model_configuration = model_configuration
        self.run_dir = output.resolve() / uuid.uuid4().hex
        self.workspace: Workspace | None = None
        self.events: list[dict[str, Any]] = []
        self.checks: list[CheckResult] = []
        self.plan = ""
        self.steps = 0
        self.model_calls = 0
        self.tokens = 0
        self.reported_tokens = 0
        self.usage_estimated = False
        self.started = 0.0
        self.status = "created"
        self.summary = ""
        self.repeated: dict[str, int] = {}

    def remaining_time(self) -> float:
        return self.config.limits.max_seconds - (time.monotonic() - self.started)

    def _workspace(self) -> Workspace:
        if self.workspace is None:
            raise HarnessError("Workspace is not initialized")
        return self.workspace

    def _checkpoint(self) -> None:
        write_json(
            self.run_dir / "checkpoint.json",
            {
                "schema_version": 1,
                "status": self.status,
                "steps": self.steps,
                "model_calls": self.model_calls,
                "plan": self.plan,
                "accounted_tokens": self.tokens,
                "reported_tokens": self.reported_tokens,
                "usage_estimated": self.usage_estimated,
                "checks": [check.to_dict() for check in self.checks],
            },
        )

    def _event(self, value: dict[str, Any]) -> None:
        event = {"step": self.steps, **value}
        self.events.append(event)
        with (self.run_dir / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, ensure_ascii=True) + "\n")
            stream.flush()
        self._checkpoint()

    def _record_plan(self, plan: str) -> None:
        self.plan = plan

    def _check(self, name: str, phase: str = "agent") -> dict[str, Any]:
        check = next((item for item in self.config.checks if item.name == name), None)
        if check is None:
            raise HarnessError(f"Unknown configured check: {name}")
        if self.remaining_time() <= 0:
            raise BudgetError("Time budget exhausted")
        workspace = self._workspace()
        workspace.assert_integrity()
        fingerprint = workspace.fingerprint()
        evidence = run_check(
            check,
            workspace.root,
            self.run_dir,
            allowed=self.allow_host_execution,
            timeout=min(self.remaining_time(), self.config.limits.command_timeout),
            output_limit=self.config.limits.max_output_bytes,
            sequence=len(self.checks) + 1,
            fingerprint=fingerprint,
            phase=phase,
        )
        integrity_error = None
        try:
            workspace.assert_integrity()
            if workspace.fingerprint() != fingerprint:
                raise HarnessError(
                    "Check changed repository files; its result is not valid evidence"
                )
        except HarnessError as exc:
            integrity_error = exc
            evidence = replace(evidence, passed=False, reason="workspace_changed")
        self.checks.append(evidence)
        self._event({"type": "check", "result": evidence.to_dict()})
        if integrity_error is not None:
            raise integrity_error
        return evidence.to_dict()

    def _read_log(self, log: str, start: int, end: int, by_characters: bool) -> dict[str, Any]:
        if not any(check.log == log for check in self.checks):
            raise HarnessError("Only logs from this run's recorded checks may be read")
        text = (self.run_dir / log).read_text(encoding="utf-8", errors="replace")
        if by_characters:
            if start > len(text):
                raise HarnessError("Character offset is beyond the end of the log")
            stop = start + end
            return {
                "log": log,
                "content": text[start:stop],
                "total_characters": len(text),
                "next_offset": stop if stop < len(text) else None,
            }
        if end < start or end - start >= 200:
            raise HarnessError("Read an inclusive range of at most 200 log lines")
        lines = text.splitlines(keepends=True)
        if start > max(1, len(lines)):
            raise HarnessError("start_line is beyond the end of the log")
        content = "".join(lines[start - 1 : end])
        return {
            "log": log,
            "total_lines": len(lines),
            "content": content[:12_000],
            "truncated": len(content) > 12_000,
        }

    def _context(self) -> str:
        workspace = self._workspace()
        latest: dict[str, Any] = {}
        baseline: dict[str, Any] = {}
        for check in self.checks:
            entry = {
                "passed": check.passed,
                "reason": check.reason,
                "log": check.log,
                "fingerprint": check.fingerprint,
                "tests_executed": check.tests_executed,
            }
            latest[check.name] = entry
            if check.phase == "baseline":
                baseline[check.name] = entry
        names = sorted(workspace.initial)
        context: dict[str, Any] = {
            "task": self.task,
            "plan": self.plan,
            "initial_files": names[:80],
            "initial_file_count": len(names),
            "file_list_truncated": len(names) > 80,
            "excluded_source_entries": workspace.omitted[:40],
            "excluded_entries_truncated": len(workspace.omitted) > 40,
            "changes": workspace.changes(),
            "current_fingerprint": workspace.fingerprint(),
            "configured_checks": [asdict(check) for check in self.config.checks],
            "baseline_checks": baseline,
            "latest_checks": latest,
            "remaining_steps": self.config.limits.max_steps - self.steps,
            "remaining_accounted_tokens": self.config.limits.max_total_tokens - self.tokens,
            "recent_observations": [],
            "omitted_observation_count": len(self.events),
        }
        maximum = self.config.limits.max_context_chars
        if len(json.dumps(context)) > maximum:
            raise BudgetError("Pinned task, plan and verification state exceed the context limit")
        recent: list[dict[str, Any]] = []
        for event in reversed(self.events):
            encoded = json.dumps(event, ensure_ascii=True)
            if len(encoded) > maximum // 3:
                candidate = {
                    "step": event["step"],
                    "type": event.get("type"),
                    "truncated": True,
                    "excerpt": encoded[: maximum // 3 - 250],
                    "notice": "Request narrower file/log reads for omitted evidence.",
                }
            else:
                candidate = event
            context["recent_observations"] = [candidate, *recent]
            context["omitted_observation_count"] = len(self.events) - len(recent) - 1
            if len(json.dumps(context)) > maximum:
                break
            recent.insert(0, candidate)
        context["recent_observations"] = recent
        context["omitted_observation_count"] = len(self.events) - len(recent)
        return json.dumps(context, ensure_ascii=True)

    def _call_limits(self, context: str) -> tuple[CallLimits, int]:
        reserve_seconds = min(10.0, self.config.limits.max_seconds / 5) if self.config.checks else 0
        available_time = self.remaining_time() - reserve_seconds
        if available_time <= 0:
            raise BudgetError("Time budget reached; final-verification reserve preserved")
        # UTF-8 bytes plus framing conservatively estimate unknown model tokenization.
        input_estimate = (
            len(SYSTEM_PROMPT.encode())
            + len(context.encode())
            + len(json.dumps(TOOLS).encode())
            + 1024
        )
        available_tokens = self.config.limits.max_total_tokens - self.tokens - input_estimate
        if available_tokens < 256:
            raise BudgetError("Insufficient token budget for another bounded model request")
        maximum_output = min(self.config.limits.max_output_tokens, available_tokens)
        return CallLimits(
            maximum_output, min(60.0, available_time)
        ), input_estimate + maximum_output

    def _finish(self) -> bool:
        if not self.config.checks:
            self.status = "unverified"
            return True
        self.status = "verifying"
        results = [self._check(check.name, "final") for check in self.config.checks]
        if all(result["passed"] for result in results):
            self.status = "verified"
            return True
        self.status = "running"
        self._event(
            {
                "type": "finish_rejected",
                "error": "Final checks failed. Diagnose the failure before requesting finish.",
            }
        )
        return False

    def _loop(self) -> None:
        workspace = self._workspace()
        tools = Tools(workspace, self._check, self._record_plan, self._read_log)
        transient_errors = 0
        action_errors = 0
        for step in range(1, self.config.limits.max_steps + 1):
            self.steps = step
            workspace.assert_integrity()
            context = self._context()
            limits, reservation = self._call_limits(context)
            self.model_calls += 1
            self._event({"type": "model_request", "max_output_tokens": limits.max_output_tokens})
            try:
                reply = self.model.choose_action(SYSTEM_PROMPT, context, TOOLS, limits)
            except ModelActionError as exc:
                self.tokens += exc.input_tokens + exc.output_tokens
                self.reported_tokens += exc.input_tokens + exc.output_tokens
                self._event({"type": "model_action_error", "error": str(exc)})
                action_errors += 1
                if self.tokens > self.config.limits.max_total_tokens or self.remaining_time() <= 0:
                    raise BudgetError("Budget exhausted while correcting model output") from exc
                if action_errors >= 3:
                    raise ModelError("Model returned three unusable action responses.") from exc
                continue
            except ModelError as exc:
                self.tokens += reservation
                self.usage_estimated = True
                self._event({"type": "model_error", "error": str(exc), "retryable": exc.retryable})
                transient_errors += 1
                if not exc.retryable or transient_errors > 2:
                    raise
                delay = min(2 ** (transient_errors - 1), max(0, self.remaining_time()))
                time.sleep(delay)
                continue
            transient_errors = 0
            action_errors = 0
            if (
                type(reply.input_tokens) is not int
                or type(reply.output_tokens) is not int
                or reply.input_tokens < 0
                or reply.output_tokens < 0
            ):
                raise ModelError("Provider returned invalid usage accounting")
            usage = reply.input_tokens + reply.output_tokens
            self.tokens += usage
            self.reported_tokens += usage
            if self.tokens > self.config.limits.max_total_tokens or self.remaining_time() <= 0:
                raise BudgetError("Budget exhausted after model request; action was not executed")
            action = reply.action
            try:
                validate_action(action)
                signature = json.dumps(
                    [asdict(action), workspace.fingerprint()],
                    sort_keys=True,
                )
                self.repeated[signature] = self.repeated.get(signature, 0) + 1
                if self.repeated[signature] >= 3:
                    raise BudgetError("Repeated identical action without repository progress")
                if action.name in ("edit_file", "delete_file") and not self.plan:
                    raise HarnessError("Record an implementation plan before modifying files")
                self._event({"type": "action_started", "action": asdict(action)})
                if action.name == "finish":
                    if not action.arguments["summary"].strip():
                        raise HarnessError("A nonempty completion summary is required")
                    self.summary = action.arguments["summary"]
                    if self._finish():
                        return
                    continue
                observation = tools.execute(action)
                workspace.assert_integrity()
                self._event(
                    {
                        "type": "tool_result",
                        "action": asdict(action),
                        "result": observation,
                    }
                )
            except BudgetError:
                raise
            except HarnessError as exc:
                self._event({"type": "tool_error", "action": asdict(action), "error": str(exc)})
        raise BudgetError("Maximum model steps reached")

    def run(self) -> dict[str, Any]:
        self.started = time.monotonic()
        self.run_dir.mkdir(parents=True, mode=0o700)
        try:
            self.status = "preflight"
            write_json(
                self.run_dir / "request.json",
                {
                    "schema_version": 1,
                    "task": self.task,
                    "source": str(self.source.resolve()),
                    "model": self.model_label,
                    "model_configuration": (
                        asdict(self.model_configuration) if self.model_configuration else None
                    ),
                    "config": asdict(self.config),
                    "host_execution": self.allow_host_execution,
                },
            )
            self._checkpoint()
            self.workspace = Workspace(self.source, self.run_dir, self.config)
            write_json(
                self.run_dir / "source-manifest.json",
                {
                    "fingerprint": self.workspace.fingerprint(),
                    "files": {
                        name: {"sha256": digest(item.content), "mode": item.mode}
                        for name, item in self.workspace.initial.items()
                    },
                    "omitted": self.workspace.omitted,
                },
            )
            self.status = "baseline"
            for check in self.config.checks:
                self._check(check.name, "baseline")
            self.status = "running"
            self._loop()
        except BudgetError as exc:
            self.status, self.summary = "budget_exhausted", str(exc)
        except KeyboardInterrupt:
            self.status, self.summary = "cancelled", "Run cancelled; no actions will be replayed"
        except (HarnessError, OSError) as exc:
            self.status = "failed"
            self.summary = (
                str(exc)
                if isinstance(exc, HarnessError)
                else ("Filesystem or process I/O failed; inspect local artifacts and permissions")
            )
        changes: list[dict[str, Any]] = []
        if self.workspace is not None:
            try:
                changes = self.workspace.export(self.run_dir)
            except (OSError, HarnessError):
                self.status, self.summary = (
                    "failed",
                    "Could not export a complete, valid change set",
                )
        result = {
            "schema_version": 1,
            "status": self.status,
            "summary": self.summary,
            "model": self.model_label,
            "model_configuration": (
                asdict(self.model_configuration) if self.model_configuration else None
            ),
            "run_dir": str(self.run_dir),
            "workspace": str(self._workspace().root) if self.workspace else None,
            "changes": changes,
            "checks": [check.to_dict() for check in self.checks],
            "steps": self.steps,
            "model_calls": self.model_calls,
            "reported_tokens": self.reported_tokens,
            "accounted_tokens": self.tokens,
            "usage_estimated": self.usage_estimated,
            "duration_seconds": round(time.monotonic() - self.started, 3),
            "verification_scope": "Configured checks on the final copy; not a correctness proof",
        }
        self._checkpoint()
        write_json(self.run_dir / "result.json", result)
        return result
