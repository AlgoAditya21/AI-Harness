from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from harness.config import load_config
from harness.contracts import HarnessError
from harness.demo import DEMO_CONFIG, DEMO_TASK, DemoModel, create_demo_repository
from harness.engine import Harness

EXIT_CODES = {"verified": 0, "failed": 1, "unverified": 3, "budget_exhausted": 4, "cancelled": 130}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        prog="ai-harness",
        description="A bounded coding-agent harness. Host execution is NOT sandboxed.",
    )
    commands = result.add_subparsers(dest="command", required=True)
    live = commands.add_parser(
        "run", help="Run a configured model-backed agent (paid API requests)"
    )
    live.add_argument("--repo", type=Path, required=True)
    task = live.add_mutually_exclusive_group(required=True)
    task.add_argument("--task")
    task.add_argument("--task-file", type=Path)
    live.add_argument(
        "--model", required=True, help="Explicit model ID available at the selected provider"
    )
    live.add_argument(
        "--provider",
        choices=("deepseek", "qwen"),
        required=True,
        help="Explicit API adapter; the credential never determines the provider",
    )
    live.add_argument(
        "--base-url",
        required=True,
        help="Required HTTPS API base URL for deepseek/qwen; no evaluation host is assumed",
    )
    live.add_argument(
        "--qwen-thinking",
        choices=("provider-default", "enabled", "disabled"),
        default="provider-default",
        help="Qwen enable_thinking control; unchanged unless selected",
    )
    live.add_argument(
        "--token-limit-parameter",
        choices=("max_tokens", "max_completion_tokens"),
        default="max_tokens",
        help="Chat Completions limit field supported by your server; default max_tokens",
    )
    live.add_argument(
        "--config", type=Path, help="JSON configuration containing trusted check argv"
    )
    live.add_argument(
        "--allow-model-upload",
        action="store_true",
        help="Authorize sending non-sensitive task and code excerpts to the selected provider",
    )
    demo = commands.add_parser("demo", help="Run an offline scripted bug fix, NOT a real model")
    for subcommand in (live, demo):
        subcommand.add_argument(
            "--output",
            type=Path,
            default=Path.home() / ".ai-harness" / "runs",
            help="Artifact parent, outside the input repository",
        )
        subcommand.add_argument(
            "--allow-host-execution",
            action="store_true",
            help="Allow configured checks on this host; only use trusted code",
        )
    initialize = commands.add_parser("init-demo", help="Create a non-sensitive example repository")
    initialize.add_argument("--destination", type=Path, required=True)
    commands.add_parser("tui", help="Launch the interactive terminal UI")
    return result


def _print_result(result: dict[str, Any]) -> int:
    print(json.dumps(result, indent=2, ensure_ascii=True))
    return EXIT_CODES[result["status"]]


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "tui":
            from harness.tui import main as tui_main

            return tui_main()
        if args.command == "init-demo":
            create_demo_repository(args.destination)
            print(json.dumps({"repository": str(args.destination.resolve())}))
            return 0
        if args.command == "demo":
            with tempfile.TemporaryDirectory(prefix="ai-harness-demo-") as temporary:
                source = Path(temporary) / "repo"
                create_demo_repository(source)
                return _print_result(
                    Harness(
                        DemoModel(),
                        source,
                        DEMO_TASK,
                        args.output,
                        DEMO_CONFIG,
                        allow_host_execution=args.allow_host_execution,
                        model_label="scripted-demo-not-an-llm",
                    ).run()
                )
        if not args.allow_model_upload:
            raise HarnessError(
                "Live mode requires --allow-model-upload. Only send code you are authorized to "
                "share with the selected provider, and never sensitive repositories."
            )
        config = load_config(args.config)
        task = args.task
        if task is None:
            with args.task_file.open(encoding="utf-8") as stream:
                task = stream.read(8001)
        from harness.chat_model import ChatModel

        model = ChatModel(
            provider=args.provider,
            model=args.model,
            base_url=args.base_url,
            qwen_thinking=args.qwen_thinking,
            token_limit_parameter=args.token_limit_parameter,
        )
        try:
            return _print_result(
                Harness(
                    model,
                    args.repo,
                    task,
                    args.output,
                    config,
                    allow_host_execution=args.allow_host_execution,
                    model_label=args.model,
                    model_configuration=model.configuration,
                ).run()
            )
        finally:
            model.close()
    except (HarnessError, OSError, UnicodeError) as exc:
        message = (
            str(exc)
            if isinstance(exc, HarnessError)
            else ("Could not read or persist a required file; check paths and permissions")
        )
        print(json.dumps({"status": "error", "error": message}), file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print(json.dumps({"status": "cancelled"}), file=sys.stderr)
        return 130
