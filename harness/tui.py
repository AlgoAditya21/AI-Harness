from __future__ import annotations

import getpass
import json
import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

# Color / Style constants for modern terminal output
BOLD = "\033[1m"
DIM = "\033[2m"
ITALIC = "\033[3m"
UNDERLINE = "\033[4m"
RESET = "\033[0m"

RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
MAGENTA = "\033[95m"
CYAN = "\033[96m"
WHITE = "\033[97m"

BG_CYAN = "\033[46m\033[30m"
BG_GREEN = "\033[42m\033[30m"
BG_RED = "\033[41m\033[37m"


def clear_screen() -> None:
    """Clear terminal screen if stdout is a tty."""
    if sys.stdout.isatty():
        os.system("clear" if os.name != "nt" else "cls")


def header() -> None:
    width = 68
    print()
    print(f"{CYAN}╭{'─' * (width - 2)}╮{RESET}")
    title = "AI Coding Harness - Terminal UI"
    subtitle = "Interactive Evidence-Driven Agent Execution & Evaluation"
    print(f"{CYAN}│{BOLD}{WHITE}{title.center(width - 2)}{RESET}{CYAN}│{RESET}")
    print(f"{CYAN}│{DIM}{WHITE}{subtitle.center(width - 2)}{RESET}{CYAN}│{RESET}")
    print(f"{CYAN}╰{'─' * (width - 2)}╯{RESET}")

    key = os.environ.get("AI_API_KEY", "")
    if key:
        masked = key[:6] + "••••" + key[-4:] if len(key) > 10 else "••••"
        key_status = f"{GREEN}● Set ({masked}, {len(key)} chars){RESET}"
    else:
        key_status = f"{YELLOW}○ Not Set (export AI_API_KEY or use option [6]){RESET}"

    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    print(f" {DIM}Python:{RESET} {py_ver}  │  {DIM}API Key:{RESET} {key_status}")
    print(f"{DIM}{'─' * width}{RESET}")


def prompt_choice(prompt_text: str, default: str = "") -> str:
    default_str = f" [{default}]" if default else ""
    try:
        val = input(f"{BOLD}{prompt_text}{default_str}:{RESET} ").strip()
        return val if val else default
    except (EOFError, KeyboardInterrupt):
        print()
        return ""


def prompt_secret(prompt_text: str) -> str:
    try:
        raw = getpass.getpass(f"{BOLD}{prompt_text}:{RESET} ").strip()
        cleaned = re.sub(r"\x1b\[[0-9;]*[a-zA-Z~]", "", raw).strip().strip("'\"")
        return cleaned
    except (EOFError, KeyboardInterrupt):
        print()
        return ""


def pause() -> None:
    try:
        input(f"\n{DIM}Press Enter to return to menu...{RESET}")
    except (EOFError, KeyboardInterrupt):
        print()


def render_result_summary(result: dict[str, Any]) -> None:
    status = result.get("status", "unknown")
    if status == "verified":
        badge = f"{BG_GREEN}  ✔ VERIFIED  {RESET}"
    else:
        badge = f"{BG_RED}  ✖ {status.upper()}  {RESET}"

    print(f"\n{badge}  {BOLD}Status:{RESET} {status.upper()}")
    print(f"{DIM}{'─' * 68}{RESET}")
    if "summary" in result and result["summary"]:
        print(f"{BOLD}Summary:{RESET} {result['summary']}")
    if "model" in result and result["model"]:
        print(f"{BOLD}Model:{RESET}   {result['model']}")
    if "duration_seconds" in result:
        duration = result["duration_seconds"]
        tokens = result.get("reported_tokens", 0)
        steps = result.get("steps", 0)
        print(f"{BOLD}Stats:{RESET}   {duration:.2f}s  │  Steps: {steps}  │  Tokens: {tokens}")

    changes = result.get("changes", [])
    if changes:
        print(f"\n{BOLD}Modified Files ({len(changes)}):{RESET}")
        for ch in changes:
            path = ch.get("path", "")
            kind = ch.get("kind", "modified")
            print(f"  {CYAN}•{RESET} {path} {DIM}({kind}){RESET}")

    checks = result.get("checks", [])
    if checks:
        print(f"\n{BOLD}Verification Checks Breakdown ({len(checks)}):{RESET}")
        for check in checks:
            phase = check.get("phase", "")
            name = check.get("name", "")
            passed = check.get("passed", False)
            exit_code = check.get("exit_code", 0)
            phase_tag = f"[{phase.upper():<8}]"
            if phase == "baseline":
                if passed:
                    status_text = f"{GREEN}✔ Clean (no initial defect){RESET}"
                else:
                    status_text = f"{BLUE}ℹ Initial defect detected (expected before fix){RESET}"
            else:
                if passed:
                    status_text = f"{GREEN}✔ Passed (exit: {exit_code}){RESET}"
                else:
                    status_text = f"{RED}✖ Failed (exit: {exit_code}){RESET}"
            print(f"  {DIM}{phase_tag}{RESET} {BOLD}{name:<8}{RESET} {status_text}")

    run_dir = result.get("run_dir")
    if run_dir:
        print(f"\n{DIM}Artifacts saved to: {run_dir}{RESET}")


def action_run_live() -> None:
    clear_screen()
    header()
    print(f"{BOLD}{CYAN}▶ Run Live Model Agent{RESET}")
    print(
        f"{DIM}Note: Live runs send task instructions and non-sensitive code to the API.{RESET}\n"
    )

    api_key = os.environ.get("AI_API_KEY", "")
    if not api_key:
        print(f"{YELLOW}Warning: AI_API_KEY is not currently exported in the environment.{RESET}")
        entered = prompt_secret("Enter your API Key (or press Enter to cancel)")
        if not entered:
            print("Cancelled.")
            pause()
            return
        os.environ["AI_API_KEY"] = entered
        api_key = entered
        print(f"{GREEN}✓ API key temporarily set for this session.{RESET}\n")

    # Select provider
    print(f"{BOLD}1. Select Provider:{RESET}")
    print(f"   {CYAN}1){RESET} deepseek (DeepSeek API)")
    print(f"   {CYAN}2){RESET} qwen (DashScope / Alibaba)")
    p_choice = prompt_choice("Select provider", "1")
    if p_choice == "2":
        provider = "qwen"
        default_model = "qwen-plus"
        default_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    else:
        provider = "deepseek"
        default_model = "deepseek-chat"
        default_url = "https://api.deepseek.com/v1"

    # Select model
    model = prompt_choice("Model ID", default_model)

    # Base URL
    base_url = prompt_choice("API Base URL", default_url)

    # Repository target
    print(f"\n{BOLD}2. Target Repository:{RESET}")
    print(f"   {CYAN}1){RESET} Create fresh demo repository - Interval bug (intervals.py)")
    print(f"   {CYAN}2){RESET} Create fresh demo repository - Calculator bug (calculator.py)")
    print(f"   {CYAN}3){RESET} Enter custom repository path")
    repo_choice = prompt_choice("Choose option", "1")

    temp_repo_dir: tempfile.TemporaryDirectory[str] | None = None
    auto_task: str | None = None
    if repo_choice == "3":
        repo_path_str = prompt_choice("Enter repository path")
        repo_path = Path(repo_path_str).expanduser().resolve()
        if not repo_path.exists():
            print(f"{RED}Error: Repository path does not exist.{RESET}")
            pause()
            return
    elif repo_choice == "2":
        from harness.demo import create_calculator_demo, CALCULATOR_TASK

        temp_repo_dir = tempfile.TemporaryDirectory(prefix="ai-harness-live-")
        repo_path = Path(temp_repo_dir.name) / "repo"
        create_calculator_demo(repo_path)
        auto_task = CALCULATOR_TASK
        print(f"{GREEN}✓ Initialized clean calculator repo at: {repo_path}{RESET}")
    else:
        from harness.demo import create_demo_repository, DEMO_TASK

        temp_repo_dir = tempfile.TemporaryDirectory(prefix="ai-harness-live-")
        repo_path = Path(temp_repo_dir.name) / "repo"
        create_demo_repository(repo_path)
        auto_task = DEMO_TASK
        print(f"{GREEN}✓ Initialized clean demo repo at: {repo_path}{RESET}")

    # Task prompt
    print(f"\n{BOLD}3. Task Prompt:{RESET}")
    if auto_task:
        print(f"   {CYAN}1){RESET} Use built-in task for selected repo")
        print(f"   {CYAN}2){RESET} Enter custom task")
        task_choice = prompt_choice("Choose option", "1")
        if task_choice == "2":
            task = prompt_choice("Enter task description") or auto_task
        else:
            task = auto_task
    else:
        print(f"   {CYAN}1){RESET} Enter task description (required for custom repo)")
        task = prompt_choice("Enter task description")
        if not task:
            print(f"{RED}Error: Task description is required for custom repositories.{RESET}")
            pause()
            return

    # Checks config
    config_path_str = prompt_choice("Checks config file", "examples/checks.json")
    config_path = Path(config_path_str).resolve()
    if not config_path.exists():
        print(f"{YELLOW}Config file {config_path_str} not found; proceeding without config.{RESET}")
        config_path_arg: Path | None = None
    else:
        config_path_arg = config_path

    # Output directory
    default_output = str(Path.home() / "ai-harness-runs")
    output_dir_str = prompt_choice("Output runs directory", default_output)
    output_dir = Path(output_dir_str).expanduser().resolve()

    print(f"\n{BOLD}Ready to execute with:{RESET}")
    print(f"  • Provider: {provider}")
    print(f"  • Model:    {model}")
    print(f"  • Base URL: {base_url}")
    print(f"  • Repo:     {repo_path}")
    print(f"  • Output:   {output_dir}")

    confirm = prompt_choice("\nProceed with execution? (y/n)", "y")
    if confirm.lower() != "y":
        print("Cancelled.")
        if temp_repo_dir:
            temp_repo_dir.cleanup()
        pause()
        return

    print(f"\n{CYAN}▶ Launching harness run... Please wait.{RESET}\n")

    from harness.chat_model import ChatModel
    from harness.config import load_config
    from harness.engine import Harness

    cfg = load_config(config_path_arg)
    chat_model = ChatModel(
        provider=provider,
        model=model,
        base_url=base_url,
        qwen_thinking="provider-default",
        token_limit_parameter="max_tokens",
    )
    try:
        harness = Harness(
            chat_model,
            repo_path,
            task,
            output_dir,
            cfg,
            allow_host_execution=True,
            model_label=model,
            model_configuration=chat_model.configuration,
        )
        res = harness.run()
        render_result_summary(res)
    except Exception as exc:
        print(f"\n{RED}Execution error: {exc}{RESET}")
    finally:
        chat_model.close()
        if temp_repo_dir:
            temp_repo_dir.cleanup()

    pause()


def action_run_demo() -> None:
    clear_screen()
    header()
    print(f"{BOLD}{CYAN}▶ Run Offline Scripted Demo (make demo){RESET}")
    print(f"{DIM}Runs a non-model scripted bug fix with full verification checks.{RESET}\n")

    output_dir = Path.cwd() / ".harness-runs"

    from harness.demo import DEMO_CONFIG, DEMO_TASK, DemoModel, create_demo_repository
    from harness.engine import Harness

    with tempfile.TemporaryDirectory(prefix="ai-harness-demo-") as temporary:
        source = Path(temporary) / "repo"
        create_demo_repository(source)
        print(f"{DIM}Running scripted demo...{RESET}")
        try:
            harness = Harness(
                DemoModel(),
                source,
                DEMO_TASK,
                output_dir,
                DEMO_CONFIG,
                allow_host_execution=True,
                model_label="scripted-demo-not-an-llm",
            )
            res = harness.run()
            render_result_summary(res)
        except Exception as exc:
            print(f"{RED}Error: {exc}{RESET}")

    pause()


def action_run_tests() -> None:
    clear_screen()
    header()
    print(f"{BOLD}{CYAN}▶ Running pytest Test Suite{RESET}\n")
    venv_python = Path.cwd() / ".venv" / "bin" / "python"
    py = str(venv_python) if venv_python.exists() else sys.executable
    subprocess.run([py, "-m", "pytest", "-v"])
    pause()


def action_run_lint() -> None:
    clear_screen()
    header()
    print(f"{BOLD}{CYAN}▶ Running Linters and Type Checkers{RESET}\n")
    venv_python = Path.cwd() / ".venv" / "bin" / "python"
    py = str(venv_python) if venv_python.exists() else sys.executable

    print(f"{BOLD}1. Ruff check:{RESET}")
    subprocess.run([py, "-m", "ruff", "check", "harness", "tests"])

    print(f"\n{BOLD}2. Ruff format check:{RESET}")
    subprocess.run([py, "-m", "ruff", "format", "--check", "harness", "tests"])

    print(f"\n{BOLD}3. MyPy type check:{RESET}")
    subprocess.run([py, "-m", "mypy", "harness"])

    pause()


def action_browse_runs() -> None:
    clear_screen()
    header()
    print(f"{BOLD}{CYAN}▶ Browse Past Harness Runs{RESET}\n")

    search_dirs = [
        Path.cwd() / ".harness-runs",
        Path.home() / "ai-harness-runs",
        Path.home() / ".ai-harness" / "runs",
    ]

    found_runs: list[tuple[Path, dict[str, Any]]] = []
    for d in search_dirs:
        if not d.exists():
            continue
        for child in sorted(d.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            if child.is_dir():
                res_file = child / "result.json"
                if res_file.exists():
                    try:
                        with res_file.open(encoding="utf-8") as f:
                            data = json.load(f)
                            found_runs.append((child, data))
                    except Exception:
                        pass

    if not found_runs:
        print(f"{YELLOW}No previous runs found.{RESET}")
        pause()
        return

    print(f"{BOLD}Found {len(found_runs)} run(s):{RESET}\n")
    for idx, (path, data) in enumerate(found_runs[:10], start=1):
        status = data.get("status", "unknown")
        color = GREEN if status == "verified" else RED
        model = data.get("model", "unknown")
        duration = data.get("duration_seconds", 0.0)
        print(
            f"  {CYAN}{idx:2d}){RESET} {color}{status.upper():10s}{RESET} "
            f"Model: {model:<22} ({duration:.2f}s) │ {path.name}"
        )

    choice = prompt_choice("\nSelect a run number to inspect (or Enter to cancel)")
    if not choice.isdigit():
        return

    index = int(choice) - 1
    if 0 <= index < len(found_runs):
        clear_screen()
        header()
        run_path, data = found_runs[index]
        print(f"{BOLD}Inspecting Run: {run_path}{RESET}")
        render_result_summary(data)

        # Show patch if available
        patch_file = run_path / "changes.patch"
        if patch_file.exists():
            try:
                patch_text = patch_file.read_text(encoding="utf-8").strip()
                if patch_text:
                    print(f"\n{BOLD}Git Patch (changes.patch):{RESET}")
                    print(f"{CYAN}{patch_text}{RESET}")
            except Exception:
                pass

        pause()


def action_manage_key() -> None:
    clear_screen()
    header()
    print(f"{BOLD}{CYAN}▶ Manage API Key{RESET}\n")
    current_key = os.environ.get("AI_API_KEY", "")
    if current_key:
        masked = current_key[:6] + "••••" + current_key[-4:] if len(current_key) > 10 else "••••"
        print(f"Current key: {GREEN}{masked}{RESET} (Length: {len(current_key)})")
    else:
        print(f"Current key: {YELLOW}Not set{RESET}")

    new_key = prompt_secret("\nEnter new API key (or press Enter to keep current)")
    if new_key:
        os.environ["AI_API_KEY"] = new_key
        msg = f"✓ AI_API_KEY updated for this session ({len(new_key)} chars)."
        print(f"{GREEN}{msg}{RESET}")
    pause()


def main(argv: Sequence[str] | None = None) -> int:
    while True:
        clear_screen()
        header()
        print(f"{BOLD}Main Menu:{RESET}")
        print(f"  {CYAN}[1]{RESET} 🎯 Run Live Model Agent {DIM}(DeepSeek / Qwen){RESET}")
        print(f"  {CYAN}[2]{RESET} 🚀 Run Scripted Offline Demo {DIM}(make demo){RESET}")
        print(f"  {CYAN}[3]{RESET} 🧪 Run Test Suite {DIM}(pytest, 324 tests){RESET}")
        print(f"  {CYAN}[4]{RESET} 🔍 Run Linters & Type Checker {DIM}(ruff + mypy){RESET}")
        print(f"  {CYAN}[5]{RESET} 📁 Browse Past Runs & Diffs")
        print(f"  {CYAN}[6]{RESET} 🔑 Set / Update API Key")
        print(f"  {CYAN}[0]{RESET} 🚪 Exit\n")

        choice = prompt_choice("Select an option", "1")
        if choice in ("0", "q", "exit"):
            clear_screen()
            print(f"\n{CYAN}Goodbye!{RESET}\n")
            return 0
        elif choice == "1":
            action_run_live()
        elif choice == "2":
            action_run_demo()
        elif choice == "3":
            action_run_tests()
        elif choice == "4":
            action_run_lint()
        elif choice == "5":
            action_browse_runs()
        elif choice == "6":
            action_manage_key()


if __name__ == "__main__":
    raise SystemExit(main())
