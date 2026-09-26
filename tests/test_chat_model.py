from __future__ import annotations

import copy
import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import httpx
import pytest
from openai import APIError, OpenAI

from harness.chat_model import ChatModel, validate_base_url
from harness.contracts import Action, CallLimits, ModelActionError, ModelError, ModelReply

MODEL = "explicit-evaluation-model"
BASE_URL = "https://evaluation.invalid/compatible-mode/v1"
LIMITS = CallLimits(max_output_tokens=1200, timeout_seconds=2.5)
TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "name": "read_file",
        "description": "Read a repository file.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
            "additionalProperties": False,
        },
    },
]


def function_call(**changes: Any) -> dict[str, Any]:
    return {
        "id": "call_fixture",
        "type": "function",
        "function": {"name": "read_file", "arguments": '{"path":"example.py"}'},
        **changes,
    }


def response() -> dict[str, Any]:
    return {
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "reasoning_content": "private reasoning, not an executable instruction",
                    "tool_calls": [function_call()],
                },
            }
        ],
        "usage": {
            "prompt_tokens": 31,
            "completion_tokens": 12,
            "total_tokens": 43,
            "completion_tokens_details": {"reasoning_tokens": 5},
            "prompt_cache_hit_tokens": 8,
        },
    }


@dataclass
class MockProvider:
    payload: Any = field(default_factory=response)
    requests: list[httpx.Request] = field(default_factory=list)
    status: int = 200
    content: bytes | None = None
    failure: type[httpx.RequestError] | None = None
    client: OpenAI = field(init=False)

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.failure:
            raise self.failure("private connection details", request=request)
        if self.content is not None:
            return httpx.Response(self.status, content=self.content)
        return httpx.Response(self.status, json=self.payload)


@pytest.fixture(autouse=True)
def prevent_live_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    def reject(self: httpx.HTTPTransport, request: httpx.Request) -> httpx.Response:
        pytest.fail("Provider tests must not make live API requests")

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", reject)
    monkeypatch.delenv("AI_API_KEY", raising=False)
    for name in ("DEEPSEEK_API_KEY", "QWEN_API_KEY", "DASHSCOPE_API_KEY"):
        monkeypatch.setenv(name, "unrelated-key-placeholder")
    monkeypatch.setenv("OPENAI_API_KEY", "unrelated-development-key-placeholder")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://wrong.invalid/v1")
    monkeypatch.setenv("OPENAI_ORG_ID", "unrelated-openai-organization")
    monkeypatch.setenv("OPENAI_PROJECT_ID", "unrelated-openai-project")


@pytest.fixture
def provider() -> Iterator[MockProvider]:
    provider = MockProvider()
    with OpenAI(
        api_key="unused-test-placeholder",
        base_url="https://injected.invalid/v1",
        max_retries=2,
        http_client=httpx.Client(transport=httpx.MockTransport(provider.handle)),
    ) as client:
        provider.client = client
        yield provider


@pytest.fixture(params=["deepseek", "qwen"])
def model(request: pytest.FixtureRequest, provider: MockProvider) -> ChatModel:
    return ChatModel(request.param, MODEL, BASE_URL, client=provider.client)


def test_stateless_request_conversion_and_usage(provider: MockProvider, model: ChatModel) -> None:
    original = copy.deepcopy(TOOLS)
    first = model.choose_action("rules", "first state", TOOLS, LIMITS)
    second = model.choose_action("rules", "second state", TOOLS, LIMITS)
    assert first == second == ModelReply(Action("read_file", {"path": "example.py"}), 31, 12)
    assert TOOLS == original
    assert len(provider.requests) == 2
    for request, state in zip(provider.requests, ["first state", "second state"]):
        data = json.loads(request.content)
        assert str(request.url) == BASE_URL + "/chat/completions"
        assert request.method == "POST"
        assert data == {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": "rules"},
                {"role": "user", "content": state},
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "description": TOOLS[0]["description"],
                        "parameters": TOOLS[0]["parameters"],
                    },
                }
            ],
            "tool_choice": "auto",
            "stream": False,
            "max_tokens": 1200,
        }
        assert request.extensions["timeout"] == {
            "connect": 2.5,
            "read": 2.5,
            "write": 2.5,
            "pool": 2.5,
        }
        assert "openai-organization" not in request.headers
        assert "openai-project" not in request.headers
        assert "reasoning_content" not in request.content.decode()
    assert provider.client.max_retries == 2
    assert str(provider.client.base_url) == "https://injected.invalid/v1/"


def test_zero_usage_is_distinct_from_missing_usage(
    provider: MockProvider,
    model: ChatModel,
) -> None:
    provider.payload["usage"] = {"prompt_tokens": 0, "completion_tokens": 0}
    assert model.choose_action("", "", TOOLS, LIMITS) == ModelReply(
        Action("read_file", {"path": "example.py"}),
        0,
        0,
    )


def test_nested_function_arguments_remain_intact(
    provider: MockProvider,
    model: ChatModel,
) -> None:
    arguments = {"nested": {"values": [None, True, 3, 1.25, "\u03bb"]}}
    call = provider.payload["choices"][0]["message"]["tool_calls"][0]
    call["function"]["arguments"] = json.dumps(arguments)
    assert model.choose_action("", "", TOOLS, LIMITS).action.arguments == arguments


@pytest.mark.parametrize("setting,expected", [("enabled", True), ("disabled", False)])
def test_qwen_thinking_requires_explicit_selection(
    provider: MockProvider,
    setting: str,
    expected: bool,
) -> None:
    model = ChatModel("qwen", MODEL, BASE_URL, qwen_thinking=setting, client=provider.client)
    model.choose_action("", "", TOOLS, LIMITS)
    assert json.loads(provider.requests[0].content)["enable_thinking"] is expected
    assert model.configuration.thinking == setting


@pytest.mark.parametrize("name", ["deepseek", "qwen"])
def test_token_limit_field_is_explicit(provider: MockProvider, name: str) -> None:
    model = ChatModel(
        name,
        MODEL,
        BASE_URL,
        token_limit_parameter="max_completion_tokens",
        client=provider.client,
    )
    model.choose_action("", "", TOOLS, LIMITS)
    body = json.loads(provider.requests[0].content)
    assert body["max_completion_tokens"] == LIMITS.max_output_tokens
    assert "max_tokens" not in body


@pytest.mark.parametrize(
    "options",
    [
        {"provider": "unsupported"},
        {"provider": None},
        {"model": ""},
        {"model": None},
        {"qwen_thinking": "guess"},
        {"token_limit_parameter": "arbitrary"},
        {"provider": "deepseek", "qwen_thinking": "enabled"},
    ],
)
def test_bad_configuration_makes_no_request(
    provider: MockProvider,
    options: dict[str, Any],
) -> None:
    values = {"provider": "qwen", "model": MODEL, "base_url": BASE_URL, **options}
    with pytest.raises(ModelError):
        ChatModel(**values, client=provider.client)
    assert not provider.requests


@pytest.mark.parametrize(
    "value",
    [
        "",
        " ",
        None,
        "http://evaluation.invalid/v1",
        "file:///tmp/provider",
        "https://user:password@evaluation.invalid/v1",
        "https://user@evaluation.invalid/v1",
        "https://evaluation.invalid/v1?api_key=secret",
        "https://evaluation.invalid/v1?",
        "https://evaluation.invalid/v1#secret",
        "https://evaluation.invalid/v1#",
        "https://evaluation.invalid\\@wrong.invalid/v1",
        "https://evaluation.invalid/v1/../other",
        "https://evaluation.invalid/%2e%2e",
        "https://evaluation.invalid:invalid/v1",
        "https://evaluation.invalid:70000/v1",
        "https://evaluation.invalid:0/v1",
        "https://",
        "https://[broken/v1",
        " https://evaluation.invalid/v1",
        "https://evaluation.invalid/\nv1",
        "https://evaluation.invalid/v1/chat/completions",
        "https://evaluation.invalid/v1/responses/",
    ],
)
def test_invalid_base_url_fails_before_credentials(value: Any) -> None:
    with pytest.raises(ModelError) as caught:
        ChatModel("qwen", MODEL, value)
    assert "secret" not in str(caught.value)
    assert "password" not in str(caught.value)


@pytest.mark.parametrize(
    "value",
    [
        "https://api.deepseek.com",
        "https://api.deepseek.com/v1/",
        "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        "https://evaluation.invalid:8443/v1",
        "https://[::1]:8443/v1",
    ],
)
def test_valid_base_urls(value: str) -> None:
    assert validate_base_url(value) == value.rstrip("/")


@pytest.mark.parametrize("name", ["deepseek", "qwen"])
@pytest.mark.parametrize("key", [None, "", " "])
def test_ai_api_key_is_required_without_legacy_fallbacks(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    key: str | None,
) -> None:
    if key is not None:
        monkeypatch.setenv("AI_API_KEY", key)
    with pytest.raises(ModelError, match="AI_API_KEY"):
        ChatModel(name, MODEL, BASE_URL)


@pytest.mark.parametrize("name", ["deepseek", "qwen"])
def test_owned_client_uses_selected_key_and_closes_without_redirects(
    monkeypatch: pytest.MonkeyPatch,
    provider: MockProvider,
    name: str,
) -> None:
    monkeypatch.setenv("AI_API_KEY", "selected-provider-placeholder")
    transports: list[httpx.Client] = []

    def client(**options: Any) -> httpx.Client:
        transport = httpx.Client(transport=httpx.MockTransport(provider.handle), **options)
        transports.append(transport)
        return transport

    monkeypatch.setattr("harness.chat_model.DefaultHttpxClient", client)
    with ChatModel(name, MODEL, BASE_URL) as model:
        model.choose_action("", "", TOOLS, LIMITS)
        assert "api_key" not in vars(model)
        assert transports[0].follow_redirects is False
    assert transports[0].is_closed
    model.close()
    assert provider.requests[0].headers["authorization"] == "Bearer selected-provider-placeholder"
    assert not provider.client.is_closed()


def test_injected_client_stays_owned_by_caller(provider: MockProvider) -> None:
    with ChatModel("deepseek", MODEL, BASE_URL, client=provider.client) as model:
        model.close()
    assert not provider.client.is_closed()


def test_plain_text_requires_a_corrected_action(provider: MockProvider, model: ChatModel) -> None:
    provider.payload["choices"][0].update(
        {
            "finish_reason": "stop",
            "message": {"role": "assistant", "content": "private claim: all tests passed"},
        }
    )
    with pytest.raises(ModelActionError, match="exactly one function") as caught:
        model.choose_action("", "", TOOLS, LIMITS)
    assert caught.value.input_tokens == 31
    assert caught.value.output_tokens == 12
    assert caught.value.retryable is False
    assert "private" not in str(caught.value)
    assert len(provider.requests) == 1


@pytest.mark.parametrize(
    "calls",
    [
        [],
        None,
        {},
        [None],
        [function_call(), function_call()],
        [function_call(type="custom")],
        [function_call(function=None)],
        [function_call(function={"name": "unknown", "arguments": "{}"})],
    ],
)
def test_ambiguous_or_unknown_calls_are_not_executed(
    provider: MockProvider,
    model: ChatModel,
    calls: Any,
) -> None:
    provider.payload["choices"][0]["message"]["tool_calls"] = calls
    with pytest.raises(ModelActionError):
        model.choose_action("", "", TOOLS, LIMITS)
    assert len(provider.requests) == 1


@pytest.mark.parametrize(
    "arguments",
    [
        {},
        None,
        "",
        "private non-JSON text",
        "[]",
        "null",
        "true",
        '{"path":"a","path":"b"}',
        '{"x":NaN}',
        '{"x":Infinity}',
        '{"x":1e999}',
        '{"nested":{"a":1,"a":2}}',
        '{"path":"a"} trailing',
    ],
)
def test_invalid_json_arguments_are_rejected_with_real_usage(
    provider: MockProvider,
    model: ChatModel,
    arguments: Any,
) -> None:
    call = provider.payload["choices"][0]["message"]["tool_calls"][0]
    call["function"]["arguments"] = arguments
    with pytest.raises(ModelActionError) as caught:
        model.choose_action("", "", TOOLS, LIMITS)
    assert caught.value.input_tokens == 31
    assert "private" not in str(caught.value)


def test_text_alongside_tool_call_is_not_treated_as_an_action(
    provider: MockProvider,
    model: ChatModel,
) -> None:
    provider.payload["choices"][0]["message"]["content"] = "I will read the file."
    assert model.choose_action("", "", TOOLS, LIMITS).action.name == "read_file"


def test_truncated_tool_call_is_never_accepted(provider: MockProvider, model: ChatModel) -> None:
    provider.payload["choices"][0]["finish_reason"] = "length"
    with pytest.raises(ModelActionError, match="output limit"):
        model.choose_action("", "", TOOLS, LIMITS)


@pytest.mark.parametrize("finish_reason", [None, "in_progress", "unknown"])
def test_incomplete_response_is_fatal(
    provider: MockProvider,
    model: ChatModel,
    finish_reason: Any,
) -> None:
    provider.payload["choices"][0]["finish_reason"] = finish_reason
    with pytest.raises(ModelError, match="completed response") as caught:
        model.choose_action("", "", TOOLS, LIMITS)
    assert not isinstance(caught.value, ModelActionError)


@pytest.mark.parametrize("refusal", [True, False])
def test_filtered_and_refused_responses_are_not_retried(
    provider: MockProvider,
    model: ChatModel,
    refusal: bool,
) -> None:
    if refusal:
        provider.payload["choices"][0]["message"]["refusal"] = "private refusal"
    else:
        provider.payload["choices"][0]["finish_reason"] = "content_filter"
    with pytest.raises(ModelError, match="refused") as caught:
        model.choose_action("", "", TOOLS, LIMITS)
    assert not isinstance(caught.value, ModelActionError)
    assert caught.value.retryable is False
    assert "private" not in str(caught.value)


@pytest.mark.parametrize("field", ["prompt_tokens", "completion_tokens"])
@pytest.mark.parametrize("value", [None, -1, True, False, 2.5, "3", {}, []])
def test_usage_is_not_coerced(
    provider: MockProvider,
    model: ChatModel,
    field: str,
    value: Any,
) -> None:
    provider.payload["usage"][field] = value
    with pytest.raises(ModelError, match="valid token usage"):
        model.choose_action("", "", TOOLS, LIMITS)


@pytest.mark.parametrize("usage", [None, {}, [], {"prompt_tokens": 1}, {"completion_tokens": 1}])
def test_missing_usage_is_not_fabricated(
    provider: MockProvider,
    model: ChatModel,
    usage: Any,
) -> None:
    provider.payload["usage"] = usage
    with pytest.raises(ModelError, match="valid token usage"):
        model.choose_action("", "", TOOLS, LIMITS)


@pytest.mark.parametrize(
    "content",
    [
        b"not JSON",
        b"\xff",
        b'{"usage":{},"usage":{}}',
        b'{"usage":{"prompt_tokens":NaN}}',
    ],
)
def test_invalid_wire_json_is_fatal(
    provider: MockProvider, model: ChatModel, content: bytes
) -> None:
    provider.content = content
    with pytest.raises(ModelError, match="invalid response JSON"):
        model.choose_action("", "", TOOLS, LIMITS)


@pytest.mark.parametrize("choices", [[], None, {}, [None], [{}, {}]])
def test_invalid_choices_are_rejected(
    provider: MockProvider, model: ChatModel, choices: Any
) -> None:
    provider.payload["choices"] = choices
    with pytest.raises(ModelError, match="exactly one choice"):
        model.choose_action("", "", TOOLS, LIMITS)


@pytest.mark.parametrize(
    "limits",
    [
        CallLimits(0, 1),
        CallLimits(True, 1),
        CallLimits(1, float("inf")),
        CallLimits(1, float("nan")),
        CallLimits(1, 0),
    ],
)
def test_invalid_limits_never_make_a_request(
    provider: MockProvider,
    model: ChatModel,
    limits: CallLimits,
) -> None:
    with pytest.raises(ModelError):
        model.choose_action("", "", TOOLS, limits)
    assert not provider.requests


@pytest.mark.parametrize(
    "tools",
    [
        [],
        TOOLS * 2,
        [{"type": "function", "name": "a"}],
        [{"type": "function", "name": "a", "parameters": {}, "description": False}],
        [{"type": "web_search"}],
    ],
)
def test_invalid_tool_schemas_never_make_a_request(
    provider: MockProvider,
    model: ChatModel,
    tools: list[dict[str, Any]],
) -> None:
    with pytest.raises(ModelError):
        model.choose_action("", "", tools, LIMITS)
    assert not provider.requests


@pytest.mark.parametrize(
    "status,retryable",
    [
        (400, False),
        (401, False),
        (403, False),
        (404, False),
        (422, False),
        (429, True),
        (500, True),
        (503, True),
    ],
)
def test_safe_sdk_errors_and_no_hidden_retries(
    provider: MockProvider,
    model: ChatModel,
    status: int,
    retryable: bool,
) -> None:
    provider.status = status
    provider.payload = {"error": {"message": "private content and key"}}
    with pytest.raises(ModelError, match=f"HTTP {status}") as caught:
        model.choose_action("private context", "", TOOLS, LIMITS)
    assert len(provider.requests) == 1
    assert caught.value.retryable is retryable
    assert isinstance(caught.value.__cause__, APIError)
    assert "private" not in str(caught.value)


@pytest.mark.parametrize("failure", [httpx.ReadTimeout, httpx.ConnectError])
def test_transport_failures_are_safe(
    provider: MockProvider,
    model: ChatModel,
    failure: type[httpx.RequestError],
) -> None:
    provider.failure = failure
    with pytest.raises(ModelError) as caught:
        model.choose_action("", "", TOOLS, LIMITS)
    assert caught.value.retryable is True
    assert "private" not in str(caught.value)
    assert len(provider.requests) == 1
