from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
import logging
import re
from typing import TYPE_CHECKING

from memory.models import TaskState

if TYPE_CHECKING:
    from llm.client import LLMClient

logger = logging.getLogger("memory")


class ConfirmationSignal(Enum):
    CONFIRMED = auto()
    REJECTED = auto()
    AMBIGUOUS = auto()


@dataclass(frozen=True)
class ConfirmationResult:
    signal: ConfirmationSignal
    confidence: float
    raw_answer: str


@dataclass(frozen=True)
class ConfirmationContext:
    """Контекст для бинарной классификации подтверждения в PLANNING."""

    user_message: str
    current_state: TaskState
    plan_summary: str


CONFIRMATION_PROMPT = """
Агент показал пользователю план работы и ждёт подтверждения для перехода к выполнению.

ПЛАН:
{plan_summary}

СООБЩЕНИЕ ПОЛЬЗОВАТЕЛЯ:
"{user_message}"

Твоя задача: определить намерение пользователя.

Ответь СТРОГО одним словом:
- YES — если пользователь выражает согласие, одобрение или готовность двигаться дальше
         (примеры: "да", "давай", "подходит", "окей", "норм", "пойдёт", "хорошо", "поехали")
- NO  — если пользователь задаёт вопрос, вносит правку, возражает, уточняет или меняет требования
         (примеры: "а можно добавить...", "подожди", "нет", "лучше сделай...", "а что если...")

Отвечай ТОЛЬКО одним словом: YES или NO. Без пояснений.
""".strip()


class ConfirmationClassifier:
    """
    Семантическая классификация подтверждения через LLM.
    Не использует списки слов и вызывается только в PLANNING.
    """

    def __init__(self, llm_client: "LLMClient | None", model: str):
        self._client = llm_client
        self._model = str(model or "gpt-5-nano")

    def classify(self, ctx: ConfirmationContext) -> ConfirmationResult:
        if ctx.current_state != TaskState.PLANNING:
            return ConfirmationResult(
                signal=ConfirmationSignal.AMBIGUOUS,
                confidence=0.0,
                raw_answer="skipped: not in PLANNING state",
            )
        if self._client is None:
            return ConfirmationResult(
                signal=ConfirmationSignal.AMBIGUOUS,
                confidence=0.0,
                raw_answer="skipped: llm unavailable",
            )

        prompt = CONFIRMATION_PROMPT.format(
            plan_summary=str(ctx.plan_summary or "").strip() or "План не сформирован.",
            user_message=str(ctx.user_message or "").strip(),
        )

        try:
            response = self._client.chat_completion(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=5,
                temperature=0,
            )
            raw_answer = str(response.choices[0].message.content or "").strip()
            normalized = self._normalize_binary_answer(raw_answer)

            if normalized == "YES":
                return ConfirmationResult(
                    signal=ConfirmationSignal.CONFIRMED,
                    confidence=0.95,
                    raw_answer=raw_answer,
                )
            if normalized == "NO":
                return ConfirmationResult(
                    signal=ConfirmationSignal.REJECTED,
                    confidence=0.95,
                    raw_answer=raw_answer,
                )
            return ConfirmationResult(
                signal=ConfirmationSignal.AMBIGUOUS,
                confidence=0.0,
                raw_answer=raw_answer,
            )
        except Exception as exc:
            logger.warning("[CONFIRMATION_CLASSIFIER] LLM error: %s — returning AMBIGUOUS", exc)
            return ConfirmationResult(
                signal=ConfirmationSignal.AMBIGUOUS,
                confidence=0.0,
                raw_answer=f"error: {exc}",
            )

    @staticmethod
    def _normalize_binary_answer(value: str) -> str:
        normalized = str(value or "").strip().upper().strip("`\"' .,!?:;")
        if normalized in {"YES", "NO"}:
            return normalized
        match = re.search(r"\b(YES|NO)\b", normalized)
        return str(match.group(1) if match else "")
