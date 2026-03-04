from __future__ import annotations

from typing import Any, Callable

from llm.client import LLMChatResponse, LLMChoice, LLMMessage, LLMUsage


class MockLLMClient:
    """Deterministic LLM client for tests and local demos."""

    def __init__(self, responder: Callable[[list[dict[str, str]], dict[str, Any]], str] | None = None):
        self.responder = responder or self._default_responder
        self.calls: list[dict[str, Any]] = []

    def chat_completion(self, **kwargs: Any) -> LLMChatResponse:
        messages = kwargs.get("messages") or []
        self.calls.append(kwargs)
        content = self.responder(messages, kwargs)
        prompt_chars = sum(len((m or {}).get("content", "")) for m in messages)
        usage = LLMUsage(
            prompt_tokens=max(1, prompt_chars // 4),
            completion_tokens=max(1, len(content) // 4),
            total_tokens=max(2, (prompt_chars + len(content)) // 4),
        )
        return LLMChatResponse(
            id="mock-response",
            model=str(kwargs.get("model") or "mock-model"),
            choices=[LLMChoice(message=LLMMessage(content=content), finish_reason="stop")],
            usage=usage,
            raw={"mock": True},
        )

    def _default_responder(self, messages: list[dict[str, str]], kwargs: dict[str, Any]) -> str:
        del kwargs
        last_user = ""
        for message in reversed(messages):
            if message.get("role") == "user":
                last_user = message.get("content", "")
                break
        if "код" in last_user.lower():
            return "Вот черновой ответ с кодом."
        return "Принято. Продолжаю по задаче."
