from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class LLMMessage:
    content: str


@dataclass
class LLMChoice:
    message: LLMMessage
    finish_reason: str | None = None


@dataclass
class LLMUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class LLMChatResponse:
    id: str | None
    model: str | None
    choices: list[LLMChoice]
    usage: LLMUsage
    raw: Any = None


class LLMClient(Protocol):
    def chat_completion(self, **kwargs: Any) -> LLMChatResponse:
        ...
