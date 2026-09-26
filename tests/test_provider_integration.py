from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from openai import OpenAI

from harness.chat_model import ChatModel
from harness.cli import main
from harness.config import Config
from harness.contracts import CallLimits, ModelError
from harness.demo import DemoModel, create_demo_repository
from harness.engine import Harness
from harness.model_common import runtime_api_key


@pytest.fixture(autouse=True)
def prevent_network_and_ambient_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    def reject(self: httpx.HTTPTransport, request: httpx.Request) -> httpx.Response:
        pytest.fail("Integration tests must not use real model services")

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", reject)
    monkeypatch.delenv("AI_API_KEY", raising=False)
    for variable in ("OPENAI_API_KEY", "DEEPSEEK_API_KEY", "QWEN_API_KEY", "DASHSCOPE_API_KEY"):
        monkeypatch.setenv(variable, "legacy-variable-placeholder")


@pytest.mark.parametrize("provider", ["deepseek", "qwen"])
def test_runtime_key_replacement_and_cli_e2e(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    provider: str,
) -> None:
    source = tmp_path / "repo"
    create_demo_repository(source)
    original = (source / "intervals.py").read_bytes()
    config = tmp_path / "checks.json"
    config.write_text(
        json.dumps(
            {
                "checks": [
                    {
                        "name": "tests",
                        "argv": ["{python}", "-m", "unittest", "discover", "-s", "tests", "-v"],
                        "kind": "unittest",
                    }
                ]
            }
        )
    )
    received: list[httpx.Request] = []
    transports: list[httpx.Client] = []
    scripted = DemoModel()

    def respond(request: httpx.Request) -> httpx.Response:
        received.append(request)
        body = json.loads(request.content)
        context = body["messages"][1]["content"]
        maximum = body.get("max_completion_tokens", body.get("max_tokens"))
        reply = scripted.choose_action("", context, [], CallLimits(maximum, 2))
        function = {
            "name": reply.action.name,
            "arguments": json.dumps(reply.action.arguments),
        }
        payload = {
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "reasoning_content": "discarded-provider-reasoning",
                        "tool_calls": [
                            {"id": "test-call", "type": "function", "function": function}
                        ],
                    },
                }
            ],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50},
        }
        return httpx.Response(200, json=payload)

    def http_client(**kwargs: Any) -> httpx.Client:
        client = httpx.Client(transport=httpx.MockTransport(respond), **kwargs)
        transports.append(client)
        return client

    monkeypatch.setattr("harness.chat_model.DefaultHttpxClient", http_client)
    args = [
        "run",
        "--repo",
        str(source),
        "--task",
        "Fix closed-interval endpoint handling.",
        "--provider",
        provider,
        "--base-url",
        "https://evaluator.invalid/v1",
        "--model",
        "selected-fixture-model",
        "--config",
        str(config),
        "--output",
        str(tmp_path / "runs"),
        "--allow-model-upload",
        "--allow-host-execution",
    ]
    if provider == "qwen":
        args += ["--qwen-thinking", "disabled", "--token-limit-parameter", "max_completion_tokens"]

    for marker in ("development-credential-placeholder", "evaluator-credential-placeholder"):
        monkeypatch.setenv("AI_API_KEY", marker)
        scripted = DemoModel()
        received.clear()
        assert main(args) == 0
        captured = capsys.readouterr()
        result = json.loads(captured.out)
        assert result["status"] == "verified"
        assert result["model_configuration"]["provider"] == provider
        assert result["model"] == "selected-fixture-model"
        assert result["model_calls"] == len(received) == 6
        assert result["reported_tokens"] == 900
        assert result["checks"][-1]["tests_executed"] == 7
        assert all(request.headers["authorization"] == f"Bearer {marker}" for request in received)
        assert all(request.url.path == "/v1/chat/completions" for request in received)
        assert marker not in captured.out + captured.err
        directory = Path(result["run_dir"])
        stored = json.loads((directory / "request.json").read_text())
        assert stored["model_configuration"] == result["model_configuration"]
        for artifact in directory.rglob("*"):
            if artifact.is_file() and artifact.suffix in (".json", ".jsonl", ".log"):
                assert marker not in artifact.read_text()
        assert all(client.is_closed for client in transports)
    assert (source / "intervals.py").read_bytes() == original


@pytest.mark.parametrize("provider", ["deepseek", "qwen"])
def test_missing_runtime_key_is_a_clear_cli_error(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    provider: str,
) -> None:
    args = [
        "run",
        "--repo",
        str(tmp_path),
        "--task",
        "A fixture task",
        "--provider",
        provider,
        "--base-url",
        "https://evaluator.invalid/v1",
        "--model",
        "configured-model",
        "--allow-model-upload",
    ]
    assert main(args) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "AI_API_KEY" in json.loads(captured.err)["error"]
    assert "legacy-variable-placeholder" not in captured.err


@pytest.mark.parametrize("provider", ["deepseek", "qwen"])
def test_endpoint_is_not_guessed(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    provider: str,
) -> None:
    with pytest.raises(SystemExit) as caught:
        main(
            [
                "run",
                "--repo",
                str(tmp_path),
                "--task",
                "fixture",
                "--provider",
                provider,
                "--model",
                "configured-model",
                "--allow-model-upload",
            ]
        )
    assert caught.value.code == 2
    assert "--base-url" in capsys.readouterr().err


def test_removed_provider_is_rejected(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as caught:
        main(
            [
                "run",
                "--repo",
                str(tmp_path),
                "--task",
                "fixture",
                "--model",
                "configured-model",
                "--allow-model-upload",
                "--provider",
                "openai",
                "--base-url",
                "https://evaluator.invalid/v1",
            ]
        )
    assert caught.value.code == 2
    assert "invalid choice: 'openai'" in capsys.readouterr().err


@pytest.mark.parametrize("provider", ["deepseek", "qwen"])
def test_text_only_reply_is_corrected_with_billed_usage(tmp_path: Path, provider: str) -> None:
    source = tmp_path / "repo"
    create_demo_repository(source)
    states = []

    def respond(request: httpx.Request) -> httpx.Response:
        state = json.loads(json.loads(request.content)["messages"][1]["content"])
        states.append(state)
        message: dict[str, Any] = {"role": "assistant", "content": "Done without a tool"}
        finish_reason = "stop"
        if len(states) == 2:
            message["tool_calls"] = [
                {
                    "type": "function",
                    "function": {
                        "name": "finish",
                        "arguments": '{"summary":"No checks configured"}',
                    },
                }
            ]
            finish_reason = "tool_calls"
        return httpx.Response(
            200,
            json={
                "choices": [{"finish_reason": finish_reason, "message": message}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 50},
            },
        )

    with OpenAI(
        api_key="unused-test-placeholder",
        http_client=httpx.Client(transport=httpx.MockTransport(respond)),
    ) as client:
        model = ChatModel(provider, "fixture-model", "https://evaluator.invalid/v1", client=client)
        result = Harness(
            model,
            source,
            "Inspect the repository",
            tmp_path / "runs",
            Config(),
            model_configuration=model.configuration,
        ).run()
    assert result["status"] == "unverified"
    assert result["reported_tokens"] == result["accounted_tokens"] == 300
    assert result["usage_estimated"] is False
    assert result["model_calls"] == 2
    assert any(event["type"] == "model_action_error" for event in states[-1]["recent_observations"])


def test_unusable_actions_stop_after_three_responses(tmp_path: Path) -> None:
    source = tmp_path / "repo"
    create_demo_repository(source)

    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "plain text"},
                    }
                ],
                "usage": {"prompt_tokens": 100, "completion_tokens": 50},
            },
        )

    with OpenAI(
        api_key="unused-test-placeholder",
        http_client=httpx.Client(transport=httpx.MockTransport(respond)),
    ) as client:
        model = ChatModel("qwen", "fixture-model", "https://evaluator.invalid/v1", client=client)
        result = Harness(model, source, "Inspect", tmp_path / "runs", Config()).run()
    assert result["status"] == "failed"
    assert result["model_calls"] == 3
    assert result["reported_tokens"] == result["accounted_tokens"] == 450
    assert result["usage_estimated"] is False
    assert result["changes"] == []


@pytest.mark.parametrize(
    "value", [" ", " leading", "trailing ", "line\nbreak", "a\rb", "a\tb", "a\x7fb"]
)
def test_invalid_runtime_key_is_not_echoed(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    monkeypatch.setenv("AI_API_KEY", value)
    with pytest.raises(ModelError, match="AI_API_KEY") as caught:
        runtime_api_key()
    if value.strip():
        assert value not in str(caught.value)


def test_environment_files_are_not_loaded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("AI_API_KEY=nonfunctional-test-placeholder\n")
    with pytest.raises(ModelError, match="AI_API_KEY"):
        runtime_api_key()


def test_shared_credential_requires_explicit_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("AI_API_KEY", "runtime-only-placeholder")
    with pytest.raises(SystemExit) as caught:
        main(
            [
                "run",
                "--repo",
                str(tmp_path),
                "--task",
                "fixture",
                "--model",
                "selected-model",
                "--base-url",
                "https://evaluator.invalid/v1",
                "--allow-model-upload",
            ]
        )
    assert caught.value.code == 2
    error = capsys.readouterr().err
    assert "--provider" in error
    assert "runtime-only-placeholder" not in error
