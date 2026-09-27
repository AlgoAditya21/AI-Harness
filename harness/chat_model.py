from __future__ import annotations

from types import TracebackType
from typing import Any, cast
from urllib.parse import urlsplit

from openai import APIError, APIStatusError, DefaultHttpxClient, OpenAI
from openai.types.chat import ChatCompletionToolParam
from openai.types.chat.completion_create_params import CompletionCreateParamsNonStreaming

from harness.contracts import (
    Action,
    CallLimits,
    ModelActionError,
    ModelConfiguration,
    ModelError,
    ModelReply,
)
from harness.model_common import (
    decode_json,
    parse_usage,
    runtime_api_key,
    tool_names,
    transient_api_error,
    validate_limits,
)


def validate_base_url(value: str) -> str:
    message = "Provide an explicit HTTPS API base URL without credentials, query, or fragment."
    if (
        not isinstance(value, str)
        or not value
        or any(ord(character) <= 32 or ord(character) >= 127 for character in value)
    ):
        raise ModelError(message)
    try:
        url = urlsplit(value)
        port = url.port
        if (
            url.scheme != "https"
            or not url.hostname
            or url.username is not None
            or url.password is not None
            or url.query
            or url.fragment
            or "?" in value
            or "#" in value
            or "\\" in value
            or "%" in value
            or (port is not None and port == 0)
        ):
            raise ValueError("Invalid endpoint")
        if any(part in (".", "..") for part in url.path.split("/")):
            raise ValueError("Invalid endpoint path")
    except ValueError as exc:
        raise ModelError(message) from exc
    base_url = value.rstrip("/")
    if base_url.endswith(("/chat/completions", "/responses")):
        raise ModelError("Use the API base URL, not the /chat/completions or /responses endpoint.")
    return base_url


def chat_tools(tools: list[dict[str, Any]]) -> list[ChatCompletionToolParam]:
    tool_names(tools)
    wrapped: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool.get("parameters"), dict):
            raise ModelError("Each function needs an object parameter schema.")
        function = {"name": tool["name"], "parameters": tool["parameters"]}
        if "description" in tool:
            if not isinstance(tool["description"], str):
                raise ModelError("Function descriptions must be strings.")
            function["description"] = tool["description"]
        # Strict-mode support differs by server. Arguments remain validated locally.
        wrapped.append({"type": "function", "function": function})
    return cast("list[ChatCompletionToolParam]", wrapped)


class ChatModel:
    """Stateless Chat Completions for explicitly configured DeepSeek/Qwen hosts."""

    def __init__(
        self,
        provider: str,
        model: str,
        base_url: str,
        *,
        qwen_thinking: str = "provider-default",
        token_limit_parameter: str = "max_tokens",
        client: OpenAI | None = None,
    ) -> None:
        if provider not in ("deepseek", "qwen"):
            raise ModelError("Chat provider must be deepseek or qwen.")
        if not isinstance(model, str) or not model.strip():
            raise ModelError("A model name must be configured.")
        base_url = validate_base_url(base_url)
        if qwen_thinking not in ("provider-default", "enabled", "disabled"):
            raise ModelError("Qwen thinking must be provider-default, enabled, or disabled.")
        if provider != "qwen" and qwen_thinking != "provider-default":
            raise ModelError("The Qwen thinking option is supported only by the qwen provider.")
        if token_limit_parameter not in ("max_tokens", "max_completion_tokens"):
            raise ModelError("Unsupported Chat Completions token-limit parameter.")
        self._model = model
        self.configuration = ModelConfiguration(
            provider,
            "chat-completions",
            base_url,
            qwen_thinking,
            token_limit_parameter,
        )
        self._owns_client = client is None
        if client is None:
            client = OpenAI(
                api_key=runtime_api_key(),
                base_url=base_url,
                max_retries=0,
                http_client=DefaultHttpxClient(follow_redirects=False),
            )
        self._client = client.with_options(
            base_url=base_url,
            max_retries=0,
        )
        # The SDK re-reads OpenAI environment defaults when copying a client.
        self._client.organization = None
        self._client.project = None

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> ChatModel:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def choose_action(
        self,
        instructions: str,
        context: str,
        tools: list[dict[str, Any]],
        limits: CallLimits,
    ) -> ModelReply:
        timeout = validate_limits(limits)
        names = tool_names(tools)
        wrapped = chat_tools(tools)
        settings = self.configuration
        extra_body = None
        if settings.provider == "qwen" and settings.thinking != "provider-default":
            extra_body = {"enable_thinking": settings.thinking == "enabled"}
        parameters: CompletionCreateParamsNonStreaming = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": instructions},
                {"role": "user", "content": context},
            ],
            "tools": wrapped,
            "tool_choice": "auto",
            "stream": False,
            "parallel_tool_calls": False,
        }
        if settings.token_limit_parameter == "max_tokens":
            parameters["max_tokens"] = limits.max_output_tokens
        else:
            parameters["max_completion_tokens"] = limits.max_output_tokens
        try:
            raw = self._client.chat.completions.with_raw_response.create(
                **parameters,
                extra_body=extra_body,
                timeout=timeout,
            )
        except APIError as exc:
            status = f" (HTTP {exc.status_code})" if isinstance(exc, APIStatusError) else ""
            raise ModelError(
                f"The {settings.provider} request failed{status}.",
                retryable=transient_api_error(exc),
            ) from exc
        response = decode_json(
            raw.http_response.content,
            f"{settings.provider} returned invalid response JSON.",
        )
        if not isinstance(response, dict):
            raise ModelError("Chat Completions response must be an object.")
        input_tokens, output_tokens = parse_usage(
            response.get("usage"),
            "prompt_tokens",
            "completion_tokens",
            settings.provider,
        )
        choices = response.get("choices")
        if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
            raise ModelError("Chat Completions must return exactly one choice.")
        choice = choices[0]
        message = choice.get("message")
        if not isinstance(message, dict) or message.get("role") != "assistant":
            raise ModelError("Chat Completions did not return an assistant message.")
        if choice.get("finish_reason") == "content_filter" or message.get("refusal"):
            raise ModelError("The provider refused the request; no action was executed.")
        if choice.get("finish_reason") == "length":
            raise ModelActionError(
                "Response hit the output limit; return one shorter complete function call.",
                input_tokens,
                output_tokens,
            )
        if choice.get("finish_reason") not in ("stop", "tool_calls"):
            raise ModelError("Chat Completions did not return a completed response.")
        calls = message.get("tool_calls")
        if choice["finish_reason"] != "tool_calls" or not isinstance(calls, list) or not calls:
            raise ModelActionError(
                "Return exactly one function call from the supplied tools. "
                "Use finish for completion, not plain text. Do not batch calls.",
                input_tokens,
                output_tokens,
            )
        # Use only the first call when the model batches multiple calls.
        call_raw = calls[0]
        if not isinstance(call_raw, dict):
            raise ModelActionError(
                "Return exactly one function call from the supplied tools. "
                "Use finish for completion, not plain text. Do not batch calls.",
                input_tokens,
                output_tokens,
            )
        call = call_raw
        if call.get("type") != "function" or not isinstance(call.get("function"), dict):
            raise ModelActionError("Select a function tool.", input_tokens, output_tokens)
        function = call["function"]
        name, arguments = function.get("name"), function.get("arguments")
        if not isinstance(name, str) or name not in names:
            raise ModelActionError(
                "Select a known function from the supplied tools.",
                input_tokens,
                output_tokens,
            )
        if not isinstance(arguments, str):
            raise ModelActionError(
                "Function arguments must be a JSON object string.",
                input_tokens,
                output_tokens,
            )
        try:
            parsed = decode_json(arguments, "Return valid, unambiguous JSON function arguments.")
        except ModelError as exc:
            raise ModelActionError(str(exc), input_tokens, output_tokens) from exc
        if not isinstance(parsed, dict):
            raise ModelActionError(
                "Function arguments must be a JSON object.",
                input_tokens,
                output_tokens,
            )
        return ModelReply(Action(name, parsed), input_tokens, output_tokens)
