from __future__ import annotations

import ast
from collections import defaultdict, deque
from datetime import datetime
import json
import logging
import re
from typing import TYPE_CHECKING

import storage
from memory.confirmation_classifier import (
    ConfirmationClassifier,
    ConfirmationContext,
    ConfirmationResult,
    ConfirmationSignal,
)
from memory.intents import IntentDecision, IntentName, parse_client_intent
from memory.long_term import LongTermMemory
from memory.models import MemoryWriteEvent, ProfileSource, TaskContext, TaskState
from memory.prompt_builder import PromptBuilder
from memory.router import MemoryRouter
from memory.short_term import ShortTermMemory
from memory.working import WorkingMemory

if TYPE_CHECKING:
    from llm.client import LLMClient

logger = logging.getLogger("memory")

PREVIEW_LENGTH = 120
MAX_DEBUG_TEXT_CHARS = 4000
MAX_WRITE_EVENTS = 50
CRITICAL_INTENTS_NO_BUDGET: frozenset[IntentName] = frozenset(
    {
        IntentName.START_EXECUTION,
        IntentName.PLAN_APPROVED,
        IntentName.YES_CONFIRMATION,
        IntentName.NO_CONFIRMATION,
        IntentName.VALIDATION_CHECKLIST_REQUEST,
        IntentName.VALIDATION_REJECT,
        IntentName.VALIDATION_CONFIRM,
    }
)

STACK_ID_LABELS: dict[str, str] = {
    "SWIFTUI": "SwiftUI",
    "SWIFTDATA": "SwiftData",
    "UIKIT": "UIKit",
    "MVVM": "MVVM",
    "REACT_NATIVE": "React Native",
    "FLUTTER": "Flutter",
    "XAMARIN": "Xamarin",
    "KOTLIN_MULTIPLATFORM": "Kotlin Multiplatform",
}


class MemoryManager:
    """Координатор трёх слоёв памяти и stateful-логики рабочего контекста.

    Менеджер объединяет short-term, working и long-term память, а также
    отвечает за синхронизацию профиля, debug-операции и гейт переходов
    между этапами state machine.
    """

    VALIDATION_CONFIRMED_SIGNAL = "__VALIDATION_CONFIRMED__"

    def __init__(
        self,
        short_term_limit: int = 30,
        *,
        llm_client: "LLMClient | None" = None,
        step_parser_model: str = "gpt-5-nano",
    ):
        """Инициализирует менеджер памяти и зависимые компоненты.

        Args:
            short_term_limit: Максимум сообщений в short-term слое на сессию.
            llm_client: Клиент LLM для intent-классификации и роутинга.
            step_parser_model: Модель для служебных intent-запросов.
        """
        self.llm_client = llm_client
        self.intent_model = str(step_parser_model or "gpt-5-nano")
        self.short_term = ShortTermMemory(limit_n=short_term_limit)
        self.working = WorkingMemory()
        self.long_term = LongTermMemory()
        self.router = MemoryRouter(llm_client=llm_client, step_parser_model=step_parser_model)
        self.prompt_builder = PromptBuilder()
        self.confirmation_classifier = ConfirmationClassifier(llm_client=llm_client, model=self.intent_model)
        self._write_events: dict[str, deque[dict]] = defaultdict(lambda: deque(maxlen=MAX_WRITE_EVENTS))
        self._intent_cache: dict[tuple[str, str], bool] = {}
        self._intent_status_cache: dict[tuple[str, str, str], IntentDecision] = {}
        self._requested_stack_cache: dict[str, str | None] = {}
        self._locked_stack_cache: dict[str, str | None] = {}
        self._aux_llm_limit = 12
        self._aux_llm_used = 0
        self.router.set_aux_llm_budget_reserver(self._reserve_aux_llm_slot)

    def begin_turn_aux_budget(self, *, limit: int = 12) -> None:
        """Сбрасывает лимит вспомогательных LLM-вызовов на текущий ход."""
        self._aux_llm_limit = max(1, int(limit))
        self._aux_llm_used = 0

    def _reserve_aux_llm_slot(self, purpose: str) -> bool:
        """Резервирует слот под служебный LLM-вызов в рамках текущего хода."""
        if self._aux_llm_used >= self._aux_llm_limit:
            logger.info(
                "[AUX_LLM_BUDGET] skipped purpose=%s used=%s limit=%s",
                str(purpose or ""),
                self._aux_llm_used,
                self._aux_llm_limit,
            )
            return False
        self._aux_llm_used += 1
        logger.info(
            "[AUX_LLM_BUDGET] consume purpose=%s used=%s/%s",
            str(purpose or ""),
            self._aux_llm_used,
            self._aux_llm_limit,
        )
        return True

    def route_user_message(
        self,
        *,
        session_id: str,
        user_id: str,
        user_message: str,
        client_intent: dict | None = None,
    ):
        """Маршрутизирует сообщение пользователя и фиксирует write-события.

        Args:
            session_id: Идентификатор активной сессии.
            user_id: Идентификатор пользователя.
            user_message: Текст сообщения пользователя.

        Returns:
            Список событий записи памяти, сформированный роутером.
        """
        events = self.router.route_user_message(
            session_id=session_id,
            user_id=user_id,
            user_message=user_message,
            working=self.working,
            long_term=self.long_term,
            client_intent=client_intent,
        )
        self._record_write_events(session_id=session_id, events=events, source="router")
        return events

    def build_messages(
        self,
        *,
        session_id: str,
        user_id: str,
        system_instructions: str,
        data_context: str,
        user_query: str,
    ) -> tuple[list[dict[str, str]], dict[str, str], dict]:
        """Собирает финальный контекст сообщений для запроса к LLM.

        Args:
            session_id: Идентификатор активной сессии.
            user_id: Идентификатор пользователя.
            system_instructions: Системные инструкции для модели.
            data_context: Внешний контекст данных (если есть).
            user_query: Текущий пользовательский запрос.

        Returns:
            Кортеж из:
            1) списка сообщений для LLM,
            2) preview-структуры для дебага,
            3) метаданных чтения long-term памяти.
        """
        short_term_context = self.short_term.get_context(session_id)
        working_ctx = self.working.load(session_id)
        long_term_ctx = self.long_term.retrieve(user_id=user_id, query=user_query, top_k=3)
        messages, preview = self.prompt_builder.build(
            system_instructions=system_instructions,
            data_context=data_context,
            long_term=long_term_ctx,
            working=working_ctx,
            short_term_messages=short_term_context,
            user_query=user_query,
        )
        read_meta = dict(long_term_ctx.get("read_meta") or {})
        logger.info(
            "[MEMORY_READ] layer=long_term hits=%s ids=%s reason=%s",
            int(read_meta.get("decision_hits", 0)) + int(read_meta.get("note_hits", 0)),
            (read_meta.get("decision_ids") or []) + (read_meta.get("note_ids") or []),
            "match/score",
        )
        logger.info("[MEMORY_READ] layer=working present=%s", bool(working_ctx))
        logger.info("[MEMORY_READ] layer=short_term turns=%s", len(short_term_context))
        return messages, preview, read_meta

    def append_turn(self, *, session_id: str, user_message: str, assistant_message: str) -> None:
        """Сохраняет текущий turn в short-term память.

        Args:
            session_id: Идентификатор активной сессии.
            user_message: Последняя реплика пользователя.
            assistant_message: Ответ ассистента.
        """
        self.short_term.append(session_id=session_id, role="user", content=user_message)
        self.short_term.append(session_id=session_id, role="assistant", content=assistant_message)

    def clear_session(self, session_id: str) -> None:
        """Полностью очищает short-term и working слой для сессии.

        Args:
            session_id: Идентификатор сессии для очистки.
        """
        self.short_term.clear_session(session_id)
        self.working.clear_session(session_id)
        self._write_events.pop(session_id, None)

    def clear_short_term_layer(self, *, session_id: str) -> bool:
        """Очищает short-term слой и связанный conversation context сессии.

        Args:
            session_id: Идентификатор сессии для очистки.

        Returns:
            `True`, если реально были удалены данные.
        """
        cleared_counts = storage.clear_session_conversation_context(session_id)
        cleared_keys = [key for key, count in cleared_counts.items() if int(count or 0) > 0]
        if cleared_keys:
            self._record_write_event(
                session_id=session_id,
                layer="short_term",
                keys=cleared_keys,
                operation="clear",
                source="debug_ui",
            )
        return bool(cleared_keys)

    def clear_working_layer(self, *, session_id: str) -> bool:
        """Очищает рабочий TaskContext для текущей сессии.

        Args:
            session_id: Идентификатор сессии для очистки.

        Returns:
            `True`, если до очистки рабочий контекст существовал.
        """
        had_task = self.working.load(session_id) is not None
        self.working.clear_session(session_id)
        if had_task:
            self._record_write_event(
                session_id=session_id,
                layer="working",
                keys=["task_context"],
                operation="clear",
                source="debug_ui",
            )
        return had_task

    def clear_long_term_layer(self, *, session_id: str, user_id: str) -> bool:
        """Очищает long-term слой пользователя.

        Args:
            session_id: Текущая сессия (для журналирования debug-события).
            user_id: Пользователь, чьи long-term данные очищаются.

        Returns:
            `True`, если реально были удалены данные.
        """
        cleared_counts = self.long_term.clear_user_memory(user_id=user_id)
        cleared_keys = [key for key, count in cleared_counts.items() if int(count or 0) > 0]
        if cleared_keys:
            self._record_write_event(
                session_id=session_id,
                layer="long_term",
                keys=cleared_keys,
                operation="clear",
                source="debug_ui",
            )
        return bool(cleared_keys)

    def delete_long_term_entry(self, *, session_id: str, user_id: str, entry_type: str, entry_id: int) -> bool:
        """Удаляет одну запись long-term памяти по типу и id.

        Args:
            session_id: Текущая сессия (для журналирования debug-события).
            user_id: Пользователь-владелец записи.
            entry_type: Тип записи (`decision` или `note`).
            entry_id: Идентификатор записи.

        Returns:
            `True`, если запись удалена.
        """
        normalized = str(entry_type or "").strip().lower()
        deleted = False
        if normalized == "decision":
            deleted = self.long_term.delete_decision(user_id=user_id, decision_id=int(entry_id))
            layer = "long_term.decision"
        elif normalized == "note":
            deleted = self.long_term.delete_note(user_id=user_id, note_id=int(entry_id))
            layer = "long_term.note"
        else:
            raise ValueError("entry_type must be 'decision' or 'note'")

        if deleted:
            self._record_write_event(
                session_id=session_id,
                layer=layer,
                keys=["id"],
                operation="delete",
                source="debug_ui",
                entry_id=int(entry_id),
            )
        return deleted

    def get_profile_snapshot(self, *, user_id: str, session_id: str | None = None) -> dict:
        """Возвращает snapshot canonical-профиля пользователя.

        Args:
            user_id: Идентификатор пользователя.
            session_id: Не используется, оставлен для совместимости вызова.

        Returns:
            Словарь с данными профиля пользователя.
        """
        del session_id
        return self.long_term.get_profile(user_id=user_id)

    def debug_update_profile_field(
        self,
        *,
        session_id: str,
        user_id: str,
        field: str,
        value: object,
    ) -> dict:
        """Обновляет поле профиля через debug-интерфейс.

        Args:
            session_id: Текущая сессия для debug-журналирования.
            user_id: Идентификатор пользователя.
            field: Имя изменяемого поля профиля.
            value: Новое значение поля.

        Returns:
            Актуальный snapshot профиля после изменения.
        """
        if str(field) in self.long_term.CANONICAL_FIELDS:
            self.long_term.update_profile_field(
                user_id=user_id,
                field=field,
                value=value,
                source=ProfileSource.DEBUG_MENU,
                verified=True,
            )
        else:
            self.long_term.add_profile_extra_field(
                user_id=user_id,
                field=field,
                value=value,
                source=ProfileSource.DEBUG_MENU,
            )
        self._record_write_event(
            session_id=session_id,
            layer="long_term.profile",
            keys=[str(field)],
            operation="save",
            source="debug_menu",
        )
        return self.get_profile_snapshot(user_id=user_id, session_id=session_id)

    def debug_delete_profile_field(self, *, session_id: str, user_id: str, field: str) -> dict:
        """Удаляет поле профиля через debug-интерфейс.

        Args:
            session_id: Текущая сессия для debug-журналирования.
            user_id: Идентификатор пользователя.
            field: Имя удаляемого поля.

        Returns:
            Актуальный snapshot профиля после удаления.
        """
        self.long_term.delete_profile_field(user_id=user_id, field=field)
        self._record_write_event(
            session_id=session_id,
            layer="long_term.profile",
            keys=[str(field)],
            operation="delete",
            source="debug_menu",
        )
        return self.get_profile_snapshot(user_id=user_id, session_id=session_id)

    def debug_add_profile_extra_field(
        self,
        *,
        session_id: str,
        user_id: str,
        field: str,
        value: object,
    ) -> dict:
        """Добавляет или обновляет extra-поле профиля через debug-интерфейс.

        Args:
            session_id: Текущая сессия для debug-журналирования.
            user_id: Идентификатор пользователя.
            field: Имя extra-поля.
            value: Значение extra-поля.

        Returns:
            Актуальный snapshot профиля после изменения.
        """
        self.long_term.add_profile_extra_field(
            user_id=user_id,
            field=field,
            value=value,
            source=ProfileSource.DEBUG_MENU,
        )
        self._record_write_event(
            session_id=session_id,
            layer="long_term.profile",
            keys=[str(field)],
            operation="save",
            source="debug_menu",
        )
        return self.get_profile_snapshot(user_id=user_id, session_id=session_id)

    def debug_confirm_profile_field(self, *, session_id: str, user_id: str, field: str) -> dict:
        """Помечает поле профиля подтверждённым через debug-интерфейс.

        Args:
            session_id: Текущая сессия для debug-журналирования.
            user_id: Идентификатор пользователя.
            field: Имя подтверждаемого поля.

        Returns:
            Актуальный snapshot профиля после подтверждения.
        """
        self.long_term.confirm_profile_field(user_id=user_id, field=field)
        self._record_write_event(
            session_id=session_id,
            layer="long_term.profile",
            keys=[str(field)],
            operation="save",
            source="debug_menu",
        )
        return self.get_profile_snapshot(user_id=user_id, session_id=session_id)

    def debug_resolve_profile_conflict(
        self,
        *,
        session_id: str,
        user_id: str,
        field: str,
        chosen_value: object | None = None,
        keep_existing: bool = False,
    ) -> dict:
        """Разрешает конфликт значения профиля через debug-интерфейс.

        Args:
            session_id: Текущая сессия для debug-журналирования.
            user_id: Идентификатор пользователя.
            field: Имя конфликтующего поля.
            chosen_value: Явно выбранное новое значение поля.
            keep_existing: Оставить текущее значение без замены.

        Returns:
            Актуальный snapshot профиля после разрешения конфликта.
        """
        self.long_term.resolve_profile_conflict(
            user_id=user_id,
            field=field,
            chosen_value=chosen_value,
            keep_existing=keep_existing,
        )
        self._record_write_event(
            session_id=session_id,
            layer="long_term.profile",
            keys=[str(field), "conflicts"],
            operation="save",
            source="debug_menu",
        )
        return self.get_profile_snapshot(user_id=user_id, session_id=session_id)

    def hydrate_short_term(self, session_id: str, messages: list[dict[str, str]]) -> None:
        """Гидратирует short-term память внешним набором сообщений.

        Args:
            session_id: Идентификатор сессии.
            messages: Сообщения для восстановления short-term контекста.
        """
        self.short_term.hydrate(session_id=session_id, messages=messages)

    def enforce_planning_gate(
        self,
        *,
        session_id: str,
        user_message: str,
        user_id: str | None = None,
        client_intent: dict | None = None,
    ) -> str | None:
        """Применяет правила state machine к входящему сообщению.

        Args:
            session_id: Идентификатор текущей сессии.
            user_message: Сообщение пользователя.
            user_id: Идентификатор пользователя (опционально).

        Returns:
            `None`, если сообщение допустимо для текущего состояния,
            либо строку с блокирующим пояснением/инструкцией.
        """
        ctx: TaskContext | None = self.working.load(session_id)
        if not ctx:
            return None

        raw_msg = str(user_message or "").strip()
        msg = raw_msg.lower()
        structured_intent, structured_payload = parse_client_intent(client_intent)
        status = self.working.get_step_status(ctx)
        step_index = status.get("step_index")
        total_steps = status.get("total_steps")

        if ctx.state == TaskState.PLANNING:
            stack_block = self._maybe_block_stack_switch_policy(
                session_id=session_id,
                user_id=user_id,
                ctx=ctx,
                user_message=raw_msg,
                structured_intent=structured_intent,
                structured_payload=structured_payload,
            )
            if stack_block:
                return stack_block
            plan_empty = self._plan_is_empty(ctx)
            if (structured_intent == IntentName.SKIP_MANDATORY_PLANNING or self._is_skip_request(msg)) and plan_empty:
                return "Пропуск PLANNING запрещён. Нужно пройти обязательный этап планирования."
            if plan_empty and self._is_direct_code_request(msg):
                return (
                    "Сначала нужен план задачи. "
                    "Опишите шаги или попросите: «Сформируй план задачи автоматически»."
                )
            if bool((ctx.vars or {}).get("plan_guidance_required")):
                return "Опишите шаги плана. Если хотите, я могу сформировать их автоматически."
            if plan_empty and (structured_intent == IntentName.START_EXECUTION or self._is_start_execution_request(msg)):
                return self._try_start_execution_from_planning(ctx)
            if (
                plan_empty
                or structured_intent == IntentName.PLAN_FORMATION
                or structured_intent == IntentName.GOAL_CLARIFICATION
                or self._is_plan_formation_message(msg)
                or self._is_goal_clarification_message(msg)
            ):
                return None

            if ctx.plan:
                confirmation_result = self._classify_planning_confirmation(
                    ctx=ctx,
                    user_message=raw_msg,
                    structured_intent=structured_intent,
                )
                logger.info(
                    "[CONFIRMATION_CLASSIFY] signal=%s confidence=%.2f raw=%r",
                    confirmation_result.signal.name,
                    confirmation_result.confidence,
                    confirmation_result.raw_answer,
                )
                if confirmation_result.signal == ConfirmationSignal.CONFIRMED:
                    return self._try_start_execution_from_planning(ctx)
                # Soft PLANNING mode: with an existing plan we continue through LLM response generation,
                # instead of returning a hardcoded planning-block message.
                return None
            return "Сначала сформируйте план задачи, затем можно перейти к выполнению."

        if ctx.state == TaskState.EXECUTION:
            stack_block = self._maybe_block_stack_switch_policy(
                session_id=session_id,
                user_id=user_id,
                ctx=ctx,
                user_message=raw_msg,
                structured_intent=structured_intent,
                structured_payload=structured_payload,
            )
            if stack_block:
                return stack_block
            if structured_intent == IntentName.THIRD_PARTY_DEPENDENCY_REQUEST or self._is_third_party_dependency_request(msg):
                return (
                    "Не добавляю сторонние зависимости в текущем MVP.\n"
                    "Почему: в задаче зафиксирован нативный подход без внешних SDK.\n"
                    "Вместо этого предлагаю: URLSession для сети и SwiftData для хранения."
                )
            all_steps_done = bool(ctx.current_step is None and ctx.done == ctx.plan and ctx.plan)

            if all_steps_done:
                if structured_intent == IntentName.VALIDATION_SKIP_REQUEST or self._is_validation_skip_request(msg):
                    return "Этап VALIDATION обязателен. Если есть замечания, напишите их и вернёмся к доработке."
                try:
                    self.working.request_validation(session_id)
                    return None
                except ValueError as exc:
                    return str(exc)

            if structured_intent == IntentName.VALIDATION_SKIP_REQUEST or self._is_validation_skip_request(msg):
                return "Пропуск этапа VALIDATION запрещён. Завершите текущий шаг и перейдите к проверке."

            if structured_intent == IntentName.VALIDATION_REQUEST or self._is_validation_request(msg):
                try:
                    self.working.request_validation(session_id)
                    return "Отлично, переходим к финальной проверке результата."
                except ValueError as exc:
                    return str(exc)

            if self._is_execution_allowed_message(msg, ctx.current_step or ""):
                return None
            return (
                f"Сейчас выполняется шаг {step_index}/{total_steps}: '{ctx.current_step}'. "
                "Уточните детали по этому шагу, либо сообщите, что шаг завершён."
            )

        if ctx.state == TaskState.VALIDATION:
            if structured_intent == IntentName.VALIDATION_CHECKLIST_REQUEST or self._is_validation_checklist_request(msg):
                return None
            if structured_intent == IntentName.VALIDATION_CONFIRM or self._is_validation_confirm_message(msg):
                return self.VALIDATION_CONFIRMED_SIGNAL
            if structured_intent == IntentName.VALIDATION_REJECT or self._is_validation_reject_message(msg):
                try:
                    self.working.transition_state(ctx, TaskState.EXECUTION)
                    ctx.updated_at = datetime.utcnow().isoformat()
                    self.working.save(ctx)
                except ValueError as exc:
                    return str(exc)
                return None
            logger.info(
                "[STATE_AUTO] VALIDATION -> DONE (implicit confirmation, no rejection detected)"
            )
            return self.VALIDATION_CONFIRMED_SIGNAL

        if ctx.state == TaskState.DONE:
            return None
        return None

    def get_working_view(self, *, session_id: str) -> dict:
        """Возвращает компактное представление рабочего состояния сессии.

        Args:
            session_id: Идентификатор сессии.

        Returns:
            Словарь со state, текущим шагом, прогрессом и флагом ожидания валидации.
        """
        ctx = self.working.load(session_id)
        if not ctx:
            return {
                "state": None,
                "current_step": None,
                "step_index": None,
                "total_steps": 0,
                "done": [],
                "plan": [],
                "awaiting_validation": False,
                "source": "none",
            }
        status = self.working.get_step_status(ctx)
        return {
            "state": status["state"],
            "current_step": status["current_step"],
            "step_index": status["step_index"],
            "total_steps": status["total_steps"],
            "done": list(ctx.done),
            "plan": list(ctx.plan),
            "awaiting_validation": bool(
                ctx.state == TaskState.EXECUTION
                and ctx.current_step is None
                and bool(ctx.plan)
                and ctx.done == ctx.plan
            ),
            "source": "working",
        }

    def get_working_actions(self, *, session_id: str) -> list[dict]:
        """Возвращает список доступных UI-действий для рабочего состояния.

        Args:
            session_id: Идентификатор сессии.

        Returns:
            Список действий (на текущем этапе всегда пустой).
        """
        del session_id
        return []

    def save_done_summary_to_long_term(
        self,
        *,
        session_id: str,
        user_id: str,
        task_title: str,
        summary: str,
    ) -> None:
        """Сохраняет итог завершённой задачи в long-term заметках.

        Args:
            session_id: Идентификатор сессии для write-логов.
            user_id: Идентификатор пользователя.
            task_title: Название завершённой задачи.
            summary: Сводка результата выполнения.
        """
        text = (
            f"Итоги завершённой задачи: {str(task_title or '').strip() or 'Текущая задача'}\n\n"
            f"{str(summary or '').strip()}"
        ).strip()
        if not text:
            return
        self.long_term.add_note(
            user_id=user_id,
            text=text,
            tags=["task_summary", "architecture_decisions"],
            source="assistant_confirmed",
            ttl_days=180,
        )
        self._record_write_event(
            session_id=session_id,
            layer="long_term.note",
            keys=["task_summary", "architecture_decisions"],
            operation="save",
            source="agent_done_summary",
        )

    def stats(self, *, session_id: str, user_id: str) -> dict:
        """Собирает агрегированную статистику по слоям памяти.

        Args:
            session_id: Идентификатор текущей сессии.
            user_id: Идентификатор пользователя.

        Returns:
            Структура с количеством записей, текущим working state и read/write метриками.
        """
        working = self.working.load(session_id)
        longterm = self.long_term.retrieve(user_id=user_id, query="", top_k=3)
        read_meta = dict(longterm.get("read_meta") or {})
        profile = longterm.get("profile") or {}
        has_profile = self._profile_snapshot_has_data(profile)
        return {
            "short_term_messages": len(self.short_term.get_context(session_id)),
            "working_state": working.state.value if working else None,
            "working_task_id": working.task_id if working else None,
            "longterm_profile": has_profile,
            "longterm_decisions": len(longterm.get("decisions") or []),
            "longterm_notes": len(longterm.get("notes") or []),
            "memory_read": read_meta,
            "recent_writes": len(self.get_recent_write_events(session_id=session_id, limit=10)),
        }

    def _profile_values_equal(self, *, field: str, left: object, right: object) -> bool:
        """Сравнивает значения профиля с учётом нормализации формата поля."""
        return self._normalize_profile_value(field=field, value=left) == self._normalize_profile_value(field=field, value=right)

    def _normalize_profile_value(self, *, field: str, value: object) -> object:
        """Приводит значение profile-поля к каноническому виду для сравнения."""
        if field in {"stack_tools", "hard_constraints"}:
            return self._normalize_text_list(value)
        if field in {"response_style", "user_role_level"}:
            return str(value or "").strip()
        if field == "project_context":
            return self._normalize_project_context_value(value)
        return value

    @staticmethod
    def _normalize_text_list(value: object) -> list[str]:
        """Нормализует список строк: trimming, удаление пустых и дублей."""
        if not isinstance(value, list):
            return []
        out: list[str] = []
        for item in value:
            text = str(item or "").strip()
            if text and text not in out:
                out.append(text)
        return out

    def _normalize_project_context_value(self, value: object) -> dict:
        """Нормализует структуру project_context в ожидаемый словарь."""
        if isinstance(value, dict):
            source = value
        else:
            source = {}
        goals = self._normalize_text_list(source.get("goals")) if isinstance(source, dict) else []
        decisions = self._normalize_text_list(source.get("key_decisions")) if isinstance(source, dict) else []
        return {
            "project_name": str(source.get("project_name") or "").strip() if isinstance(source, dict) else "",
            "goals": goals,
            "key_decisions": decisions,
        }

    def _has_open_conflict(
        self,
        *,
        profile: dict,
        field: str,
        existing_value: object,
        inferred_value: object,
    ) -> bool:
        """Проверяет, существует ли уже неразрешённый конфликт для пары значений поля."""
        conflicts = profile.get("conflicts") if isinstance(profile, dict) else []
        if not isinstance(conflicts, list):
            return False
        normalized_existing = self._normalize_profile_value(field=field, value=existing_value)
        normalized_inferred = self._normalize_profile_value(field=field, value=inferred_value)
        for conflict in conflicts:
            if not isinstance(conflict, dict):
                continue
            if str(conflict.get("field") or "") != field:
                continue
            conflict_existing = self._normalize_profile_value(field=field, value=conflict.get("existing_value"))
            conflict_inferred = self._normalize_profile_value(field=field, value=conflict.get("inferred_value"))
            if conflict_existing == normalized_existing and conflict_inferred == normalized_inferred:
                return True
        return False

    def _record_write_event(
        self,
        *,
        session_id: str,
        layer: str,
        keys: list[str],
        operation: str = "save",
        source: str = "router",
        entry_id: int | None = None,
    ) -> None:
        """Добавляет одно событие изменения памяти в локальный журнал сессии."""
        event = {
            "layer": layer,
            "keys": list(keys or []),
            "operation": operation,
            "source": source,
            "entry_id": entry_id,
            "timestamp": datetime.utcnow().isoformat(),
        }
        self._write_events[session_id].append(event)

    def _record_write_events(self, *, session_id: str, events: list[MemoryWriteEvent], source: str = "router") -> None:
        """Записывает пачку событий изменения памяти в локальный журнал."""
        for event in events:
            self._record_write_event(
                session_id=session_id,
                layer=event.layer,
                keys=event.keys,
                operation="save",
                source=source,
            )

    def get_recent_write_events(self, *, session_id: str, limit: int = 10) -> list[dict]:
        """Возвращает последние события записи памяти.

        Args:
            session_id: Идентификатор сессии.
            limit: Максимальное число возвращаемых событий.

        Returns:
            Список последних событий write-журнала.
        """
        events = list(self._write_events.get(session_id) or [])
        lim = max(1, int(limit))
        return events[-lim:]

    def _format_entry_for_debug(self, entry: dict) -> dict:
        """Форматирует запись long-term памяти в безопасный вид для Debug UI."""
        text = str(entry.get("text") or "")
        preview = text[:PREVIEW_LENGTH] + ("…" if len(text) > PREVIEW_LENGTH else "")
        full = text[:MAX_DEBUG_TEXT_CHARS]
        full_truncated = len(text) > MAX_DEBUG_TEXT_CHARS
        if full_truncated:
            full = full + "…"
        return {
            "id": entry.get("id"),
            "tags": list(entry.get("tags") or []),
            "created_at": entry.get("created_at", ""),
            "preview": preview,
            "full": full,
            "full_truncated": full_truncated,
            "entry_type": str(entry.get("type") or ""),
            "source": str(entry.get("source") or ""),
        }

    def debug_snapshot(
        self,
        *,
        session_id: str,
        user_id: str,
        query: str = "",
        top_k: int = 3,
    ) -> dict:
        """Возвращает снимок short-term, working и long-term слоёв.

        Args:
            session_id: Идентификатор текущей сессии.
            user_id: Идентификатор пользователя.
            query: Поисковый запрос для long-term retrieval.
            top_k: Лимит числа записей long-term на выдачу.

        Returns:
            Единый debug-снимок слоёв памяти и журнала write-событий.
        """
        top_k = max(1, min(10, int(top_k)))
        short_snapshot = self.short_term.snapshot(session_id)
        working_ctx = self.working.load(session_id)
        working_present = working_ctx is not None
        working_task = None
        if working_ctx:
            d = working_ctx.to_dict()
            working_task = {
                "task_id": d.get("task_id"),
                "task": d.get("task") or d.get("goal"),
                "goal": d.get("goal"),
                "state": d.get("state"),
                "plan": d.get("plan") or [],
                "current_step": d.get("current_step") or "",
                "done": d.get("done") or d.get("done_steps") or [],
                "done_steps": d.get("done_steps") or [],
                "open_questions": d.get("open_questions") or [],
                "artifacts": d.get("artifacts") or [],
                "vars": d.get("vars") or {},
                "updated_at": d.get("updated_at") or "",
            }
        long_ctx = self.long_term.retrieve(user_id=user_id, query=query or "", top_k=top_k)
        profile_compact = long_ctx.get("profile") or {}
        has_profile_data = self._profile_snapshot_has_data(profile_compact)
        decisions = long_ctx.get("decisions") or []
        notes = long_ctx.get("notes") or []
        decisions_top_k = [self._format_entry_for_debug(e) for e in decisions]
        notes_top_k = [self._format_entry_for_debug(e) for e in notes]
        read_meta = dict(long_ctx.get("read_meta") or {})
        return {
            "short_term": short_snapshot,
            "working": {
                "present": working_present,
                "task": working_task,
            },
            "long_term": {
                "profile": profile_compact,
                "has_profile_data": has_profile_data,
                "decisions_top_k": decisions_top_k,
                "notes_top_k": notes_top_k,
                "read_meta": read_meta,
            },
            "memory_writes": self.get_recent_write_events(session_id=session_id, limit=10),
        }

    @staticmethod
    def _profile_snapshot_has_data(profile: dict) -> bool:
        """Определяет, содержит ли snapshot профиля полезные непустые данные."""
        if not isinstance(profile, dict):
            return False
        project_context = (profile.get("project_context") or {}).get("value") or {}
        project_has_data = False
        if isinstance(project_context, dict):
            project_has_data = bool(
                str(project_context.get("project_name") or "").strip()
                or list(project_context.get("goals") or [])
                or list(project_context.get("key_decisions") or [])
            )
        return bool(
            (profile.get("stack_tools") or {}).get("value")
            or (profile.get("response_style") or {}).get("value")
            or (profile.get("hard_constraints") or {}).get("value")
            or (profile.get("user_role_level") or {}).get("value")
            or project_has_data
            or (profile.get("extra_fields") or {})
            or (profile.get("conflicts") or [])
        )

    def _is_start_execution_request(self, msg: str) -> bool:
        """Определяет, хочет ли пользователь перейти к этапу EXECUTION."""
        return self._classify_intent(
            intent=IntentName.START_EXECUTION,
            message=msg,
            fallback=False,
            guideline="User explicitly wants to begin implementation/execution now.",
        )

    def _is_direct_code_request(self, msg: str) -> bool:
        """Определяет прямой запрос кода в обход планирования отдельным intent-запросом."""
        return self._classify_intent(
            intent=IntentName.DIRECT_CODE_REQUEST,
            message=msg,
            fallback=False,
            guideline="User asks to produce immediate implementation code now, before/without planning steps.",
        )

    def _is_plan_approved_message(self, msg: str) -> bool:
        """Определяет, подтвердил ли пользователь план явно."""
        return self._classify_intent(
            intent=IntentName.PLAN_APPROVED,
            message=msg,
            fallback=False,
            guideline="User explicitly confirms the plan and approves moving forward.",
        )

    def _is_plan_formation_message(self, msg: str) -> bool:
        """Определяет, просит ли пользователь сформировать/уточнить план."""
        return self._classify_intent(
            intent=IntentName.PLAN_FORMATION,
            message=msg,
            fallback=False,
            guideline="User asks to form/build/refine the implementation plan.",
        )

    def _is_skip_request(self, msg: str) -> bool:
        """Определяет попытку пропуска обязательного этапа PLANNING."""
        return self._classify_intent(
            intent=IntentName.SKIP_MANDATORY_PLANNING,
            message=msg,
            fallback=False,
            guideline="User asks to bypass mandatory planning/questions and jump straight to final code/result.",
        )

    def _try_start_execution_from_planning(self, ctx: TaskContext) -> str | None:
        """Пытается перевести задачу из PLANNING в EXECUTION с автопланом при необходимости."""
        if not ctx.plan:
            fallback_plan = self.router._build_fallback_plan(goal=str(ctx.task or ctx.goal or "Текущая задача"))
            if not fallback_plan:
                return "Сначала сформируйте план задачи, затем можно перейти к выполнению."
            ctx.plan = list(fallback_plan)
            ctx.current_step = fallback_plan[0]

        if ctx.current_step != ctx.plan[0]:
            ctx.current_step = ctx.plan[0]

        try:
            self.working.transition_state(ctx, TaskState.EXECUTION)
            ctx.updated_at = datetime.utcnow().isoformat()
            self.working.save(ctx)
        except ValueError as exc:
            return str(exc)
        return None

    def _classify_planning_confirmation(
        self,
        *,
        ctx: TaskContext,
        user_message: str,
        structured_intent: IntentName | None,
    ) -> ConfirmationResult:
        if structured_intent in {
            IntentName.PLAN_APPROVED,
            IntentName.START_EXECUTION,
            IntentName.YES_CONFIRMATION,
        }:
            return ConfirmationResult(
                signal=ConfirmationSignal.CONFIRMED,
                confidence=1.0,
                raw_answer=f"client_intent:{structured_intent.value}",
            )
        if structured_intent in {
            IntentName.NO_CONFIRMATION,
            IntentName.GOAL_CLARIFICATION,
            IntentName.PLAN_FORMATION,
        }:
            return ConfirmationResult(
                signal=ConfirmationSignal.REJECTED,
                confidence=1.0,
                raw_answer=f"client_intent:{structured_intent.value}",
            )
        return self.confirmation_classifier.classify(
            ConfirmationContext(
                user_message=str(user_message or ""),
                current_state=ctx.state,
                plan_summary=self._build_plan_summary(ctx),
            )
        )

    @staticmethod
    def _build_plan_summary(ctx: TaskContext) -> str:
        plan = [str(step or "").strip() for step in list(ctx.plan or []) if str(step or "").strip()]
        if not plan:
            task = str(ctx.task or "").strip()
            return task or "План не сформирован."
        lines = [f"{index}. {step}" for index, step in enumerate(plan[:3], start=1)]
        if len(plan) > 3:
            lines.append("...")
        return "\n".join(lines)

    def _is_goal_clarification_message(self, msg: str) -> bool:
        """Определяет, относится ли сообщение к уточнению цели/требований задачи."""
        return self._classify_intent(
            intent=IntentName.GOAL_CLARIFICATION,
            message=msg,
            fallback=False,
            guideline="User asks to clarify/refine goal or requirements before planning continues.",
        )

    @staticmethod
    def _plan_is_empty(ctx: TaskContext) -> bool:
        """Проверяет, что план задачи отсутствует или пуст."""
        return len(list(ctx.plan or [])) == 0

    def _is_validation_request(self, msg: str) -> bool:
        """Определяет запрос пользователя на переход к проверке (VALIDATION)."""
        return self._classify_intent(
            intent=IntentName.VALIDATION_REQUEST,
            message=msg,
            fallback=False,
            guideline="User asks to enter/check validation phase.",
        )

    def _is_validation_checklist_request(self, msg: str) -> bool:
        """Определяет запрос показать чеклист на этапе VALIDATION."""
        return self._classify_intent(
            intent=IntentName.VALIDATION_CHECKLIST_REQUEST,
            message=msg,
            fallback=False,
            guideline="User asks for a final checklist while staying in VALIDATION stage.",
        )

    def is_validation_checklist_request(self, message: str) -> bool:
        """Публичная проверка checklist-intent для этапа VALIDATION."""
        return self._is_validation_checklist_request(str(message or "").strip().lower())

    def _is_validation_confirm_message(self, msg: str) -> bool:
        """Определяет явное подтверждение завершения задачи на этапе VALIDATION."""
        return self._classify_intent(
            intent=IntentName.VALIDATION_CONFIRM,
            message=msg,
            fallback=False,
            guideline="User explicitly confirms task completion on VALIDATION stage.",
        )

    def _is_validation_reject_message(self, msg: str) -> bool:
        """Определяет отказ от завершения и запрос на доработку."""
        return self._classify_intent(
            intent=IntentName.VALIDATION_REJECT,
            message=msg,
            fallback=False,
            guideline="User rejects completion and asks to return to EXECUTION for rework.",
        )

    def _is_validation_skip_request(self, msg: str) -> bool:
        """Определяет попытку пропуска обязательного этапа VALIDATION."""
        return self._classify_intent(
            intent=IntentName.VALIDATION_SKIP_REQUEST,
            message=msg,
            fallback=False,
            guideline="User requests to skip/bypass validation stage.",
        )

    def _is_execution_allowed_message(self, msg: str, current_step: str) -> bool:
        """Проверяет, допустимо ли сообщение в рамках текущего шага EXECUTION."""
        return self.router.is_execution_allowed_message(text=msg, current_step=current_step)

    def _maybe_block_stack_switch_policy(
        self,
        *,
        session_id: str,
        user_id: str | None,
        ctx: TaskContext,
        user_message: str,
        structured_intent: IntentName | None,
        structured_payload: dict,
    ) -> str | None:
        """Применяет policy-блокировку смены стека внутри текущей задачи.

        Args:
            session_id: Идентификатор сессии.
            user_id: Идентификатор пользователя (может быть `None`).
            ctx: Текущий рабочий контекст задачи.
            user_message: Сообщение пользователя в исходном виде.

        Returns:
            Текст блокировки, если policy сработал, иначе `None`.
        """
        locked_stack = self._resolve_locked_stack(user_id=user_id, ctx=ctx)
        if not locked_stack:
            return None

        is_switch_request, intent_status = self._is_stack_switch_request_with_status(
            msg=user_message,
            structured_intent=structured_intent,
        )
        requested_stack = self._extract_requested_stack(
            msg=user_message,
            structured_intent=structured_intent,
            payload=structured_payload,
        )
        unknown_but_conflicting_stack = bool(
            intent_status == "unknown"
            and requested_stack
            and not self._stack_matches_locked(locked_stack=locked_stack, requested_stack=requested_stack)
        )
        if not is_switch_request and not unknown_but_conflicting_stack:
            return None

        self._log_stack_switch_policy_block(
            session_id=session_id,
            state=ctx.state.value,
            locked_stack=locked_stack,
            requested_stack=requested_stack,
            intent_status="match" if is_switch_request else "unknown",
        )
        return self._build_stack_switch_block_message(
            locked_stack=locked_stack,
            requested_stack=requested_stack,
        )

    def _resolve_locked_stack(self, *, user_id: str | None, ctx: TaskContext) -> str | None:
        """Определяет зафиксированный стек: profile (verified) -> task context."""
        from_profile = self._resolve_locked_stack_from_profile(user_id=user_id)
        if from_profile:
            return from_profile
        return self._resolve_locked_stack_from_task_context(ctx)

    def _resolve_locked_stack_from_profile(self, *, user_id: str | None) -> str | None:
        """Извлекает зафиксированный стек из verified-полей профиля пользователя через отдельный анализ."""
        if not str(user_id or "").strip():
            return None
        profile = self.long_term.get_profile(user_id=str(user_id)) or {}
        if not isinstance(profile, dict):
            return None

        candidates: list[str] = []
        stack_payload = profile.get("stack_tools")
        stack_value = (stack_payload or {}).get("value") if isinstance(stack_payload, dict) else None
        stack_verified = bool((stack_payload or {}).get("verified")) if isinstance(stack_payload, dict) else False
        if stack_verified:
            candidates.extend(self._normalize_text_list(stack_value))

        constraints_payload = profile.get("hard_constraints")
        constraints_value = (constraints_payload or {}).get("value") if isinstance(constraints_payload, dict) else None
        constraints_verified = bool((constraints_payload or {}).get("verified")) if isinstance(constraints_payload, dict) else False
        if constraints_verified:
            candidates.extend(self._normalize_text_list(constraints_value))

        if not candidates:
            return None
        return self._extract_stack_from_text_via_llm(
            text="\n".join(candidates),
            cache_key=f"profile:{str(user_id)}",
            cache=self._locked_stack_cache,
        )

    def _resolve_locked_stack_from_task_context(self, ctx: TaskContext) -> str | None:
        """Извлекает зафиксированный стек из task context через отдельный анализ."""
        vars_payload = dict(ctx.vars or {})
        explicit_locked = str(vars_payload.get("locked_stack") or "").strip()
        if explicit_locked:
            return explicit_locked

        cache_key = f"task:{str(ctx.task_id or '')}:{str(ctx.updated_at or '')}"
        if cache_key in self._locked_stack_cache:
            return self._locked_stack_cache[cache_key]

        chunks: list[str] = [str(ctx.task or "")]
        chunks.extend(str(item or "") for item in list(ctx.plan or []))
        chunks.extend(str(item or "") for item in list(ctx.done or []))
        if ctx.current_step:
            chunks.append(str(ctx.current_step))
        extracted = self._extract_stack_from_text_via_llm(
            text="\n".join(chunks),
            cache_key=cache_key,
            cache=self._locked_stack_cache,
        )
        self._locked_stack_cache[cache_key] = extracted
        return extracted

    def _is_stack_switch_request_with_status(
        self,
        msg: str,
        *,
        structured_intent: IntentName | None,
    ) -> tuple[bool, str]:
        """Определяет запрос на смену стека и возвращает статус (`match|no_match|unknown`)."""
        decision = self._classify_intent_with_status(
            intent=IntentName.STACK_SWITCH_REQUEST,
            message=msg,
            guideline="User asks to switch the implementation stack/framework.",
            structured_intent=structured_intent,
        )
        return decision.is_match, decision.status

    def _extract_requested_stack(
        self,
        *,
        msg: str,
        structured_intent: IntentName | None,
        payload: dict,
    ) -> str | None:
        """Извлекает запрошенный пользователем целевой стек (структурированно или отдельным анализом)."""
        if structured_intent == IntentName.STACK_SWITCH_REQUEST:
            from_payload = str((payload or {}).get("requested_stack") or (payload or {}).get("stack") or "").strip()
            if from_payload:
                return from_payload

        normalized = str(msg or "").strip()
        if not normalized:
            return None
        if normalized in self._requested_stack_cache:
            return self._requested_stack_cache[normalized]
        extracted = self._extract_stack_from_text_via_llm(
            text=normalized,
            cache_key=normalized,
            cache=self._requested_stack_cache,
        )
        self._requested_stack_cache[normalized] = extracted
        return extracted

    def _extract_stack_from_text_via_llm(
        self,
        *,
        text: str,
        cache_key: str,
        cache: dict[str, str | None],
    ) -> str | None:
        normalized = str(text or "").strip()
        if not normalized:
            return None
        if cache_key in cache:
            return cache[cache_key]
        if self.llm_client is None:
            cache[cache_key] = None
            return None
        prompt = (
            "Extract canonical technology stack/framework from text.\n"
            "Return ONLY valid JSON with schema:\n"
            '{"stack_id": "", "stack_label": "", "confidence": 0.0}\n'
            "Allowed stack_id values: "
            f"{', '.join(STACK_ID_LABELS.keys())}\n"
            f"Text: {normalized}\n"
            "If stack is not explicit, return empty stack_id and stack_label."
        )
        extracted: str | None = None
        try:
            response = self.llm_client.chat_completion(
                model=self.intent_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=80,
                response_format={"type": "json_object"},
            )
            raw = str(response.choices[0].message.content or "").strip()
            payload = self._extract_first_json_object(raw)
            if isinstance(payload, dict):
                confidence = self._to_confidence(payload.get("confidence"))
                stack_id = self._normalize_stack_id(payload.get("stack_id"))
                label = str(payload.get("stack_label") or "").strip()
                if confidence >= 0.55:
                    if stack_id:
                        extracted = STACK_ID_LABELS.get(stack_id)
                    elif label:
                        extracted = label
        except Exception as exc:
            logger.warning("[STACK_EXTRACT] failed: %s", exc)
        cache[cache_key] = extracted
        return extracted

    @staticmethod
    def _build_stack_switch_block_message(*, locked_stack: str, requested_stack: str | None) -> str:
        """Формирует единый user-facing ответ для policy-блокировки смены стека."""
        lines = [f"Не могу сменить стек в рамках текущей задачи: зафиксирован {locked_stack}."]
        if str(requested_stack or "").strip():
            lines.append(f"Запрошен: {requested_stack}.")
        lines.append("Чтобы сменить стек, начните новую задачу или выполните reset текущей сессии.")
        return "\n".join(lines)

    @staticmethod
    def _stack_matches_locked(*, locked_stack: str, requested_stack: str) -> bool:
        requested_norm = str(requested_stack or "").strip().lower()
        locked_norm = str(locked_stack or "").strip().lower()
        if not requested_norm or not locked_norm:
            return False
        locked_parts = {part.strip() for part in locked_norm.split(",") if part.strip()}
        if requested_norm in locked_parts:
            return True
        return requested_norm == locked_norm

    @staticmethod
    def _normalize_stack_id(raw: object) -> str | None:
        value = str(raw or "").strip().upper()
        if not value:
            return None
        return value if value in STACK_ID_LABELS else None

    @staticmethod
    def _log_stack_switch_policy_block(
        *,
        session_id: str,
        state: str,
        locked_stack: str,
        requested_stack: str | None,
        intent_status: str,
    ) -> None:
        """Пишет единообразный лог policy-блокировки по смене стека."""
        logger.info(
            "[POLICY] event=policy_block reason=stack_switch_locked session_id=%s state=%s "
            "locked_stack=%s requested_stack=%s intent_status=%s",
            str(session_id or ""),
            str(state or ""),
            str(locked_stack or ""),
            str(requested_stack or "unknown"),
            str(intent_status or "unknown"),
        )

    def _is_third_party_dependency_request(self, msg: str) -> bool:
        """Определяет запрос на добавление сторонних SDK/библиотек."""
        return self._classify_intent(
            intent=IntentName.THIRD_PARTY_DEPENDENCY_REQUEST,
            message=msg,
            fallback=False,
            guideline="User asks to add external SDK/library dependency.",
        )

    def _classify_intent_with_status(
        self,
        *,
        intent: IntentName,
        message: str,
        guideline: str,
        structured_intent: IntentName | None = None,
    ) -> IntentDecision:
        """Классифицирует intent и возвращает типизированное решение.

        Args:
            intent: Имя классифицируемого интента.
            message: Входной текст.
            guideline: Краткая инструкция для intent-классификатора.
            structured_intent: Явный client intent (если передан).

        Returns:
            Структурированное решение `IntentDecision`.
        """
        normalized = str(message or "").strip()
        if not normalized:
            return IntentDecision.no_match(
                intent=intent,
                confidence=1.0,
                reason_code="intent_no_match_empty_message",
            )
        structured_name = structured_intent.value if structured_intent else ""
        cache_key = (intent.value, normalized, structured_name)
        if cache_key in self._intent_status_cache:
            return self._intent_status_cache[cache_key]
        if structured_intent is not None:
            if structured_intent == intent:
                result = IntentDecision.match(
                    intent=intent,
                    confidence=1.0,
                    reason_code="intent_match_client_intent",
                )
            else:
                result = IntentDecision.no_match(
                    intent=intent,
                    confidence=1.0,
                    reason_code="intent_no_match_client_intent_other",
                )
            self._intent_status_cache[cache_key] = result
            return result
        if self.llm_client is None:
            result = IntentDecision.unknown(
                intent=intent,
                reason_code="intent_unknown_unavailable",
            )
            logger.info(
                "[INTENT_CLASSIFY] intent=%s status=%s reason_code=%s",
                intent.value,
                result.status,
                result.reason_code,
            )
            self._intent_status_cache[cache_key] = result
            return result
        if intent not in CRITICAL_INTENTS_NO_BUDGET and not self._reserve_aux_llm_slot(f"intent:{intent.value}"):
            result = IntentDecision.unknown(
                intent=intent,
                reason_code="intent_unknown_budget",
            )
            logger.info(
                "[INTENT_CLASSIFY] intent=%s status=%s reason_code=%s",
                intent.value,
                result.status,
                result.reason_code,
            )
            self._intent_status_cache[cache_key] = result
            return result

        prompt = (
            "You are an intent classifier for a task state machine.\n"
            "Return ONLY valid JSON with this schema:\n"
            '{"match": true, "confidence": 0.0, "reason": "short"}\n'
            f"Intent: {intent.value}\n"
            f"Guideline: {guideline}\n"
            f"User message: {normalized}\n"
            "Set match=true only when intent is explicit and unambiguous."
        )
        try:
            response = self.llm_client.chat_completion(
                model=self.intent_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=80,
                response_format={"type": "json_object"},
            )
            logger.debug(
                "[INTENT_RAW_FULL] intent=%s full_response=%r",
                intent.value,
                getattr(response, "raw", response),
            )
            raw, payload = self._extract_response_payload(
                response=response,
                required_keys={"match", "confidence"},
            )
            if not isinstance(payload, dict):
                logger.info(
                    "[INTENT_CLASSIFY_RAW] intent=%s raw_preview=%s",
                    intent.value,
                    raw[:240].replace("\n", "\\n"),
                )
                recovery = self._recover_intent_with_separate_analysis(
                    intent=intent,
                    message=normalized,
                    guideline=guideline,
                )
                if recovery is not None:
                    self._intent_status_cache[cache_key] = recovery
                    return recovery
                result = IntentDecision.unknown(
                    intent=intent,
                    reason_code="intent_unknown_invalid_payload",
                )
                logger.info(
                    "[INTENT_CLASSIFY] intent=%s status=%s reason_code=%s",
                    intent.value,
                    result.status,
                    result.reason_code,
                )
                self._intent_status_cache[cache_key] = result
                return result
            confidence = self._to_confidence(payload.get("confidence"))
            if confidence < 0.6:
                recovery = self._recover_intent_with_separate_analysis(
                    intent=intent,
                    message=normalized,
                    guideline=guideline,
                )
                if recovery is not None:
                    self._intent_status_cache[cache_key] = recovery
                    return recovery
                result = IntentDecision.unknown(
                    intent=intent,
                    confidence=confidence,
                    reason=str(payload.get("reason") or ""),
                    reason_code="intent_unknown_low_confidence",
                )
                logger.info(
                    "[INTENT_CLASSIFY] intent=%s status=%s reason_code=%s",
                    intent.value,
                    result.status,
                    result.reason_code,
                )
                self._intent_status_cache[cache_key] = result
                return result
            if bool(payload.get("match")):
                result = IntentDecision.match(
                    intent=intent,
                    confidence=confidence,
                    reason=str(payload.get("reason") or ""),
                    reason_code="intent_match_llm",
                )
            else:
                result = IntentDecision.no_match(
                    intent=intent,
                    confidence=confidence,
                    reason=str(payload.get("reason") or ""),
                    reason_code="intent_no_match_llm",
                )
            self._intent_status_cache[cache_key] = result
            return result
        except Exception as exc:
            logger.warning("[INTENT_CLASSIFY] intent=%s failed: %s", intent.value, exc)
            result = IntentDecision.unknown(
                intent=intent,
                reason=str(exc),
                reason_code="intent_unknown_unavailable",
            )
            logger.info(
                "[INTENT_CLASSIFY] intent=%s status=%s reason_code=%s",
                intent.value,
                result.status,
                result.reason_code,
            )
            self._intent_status_cache[cache_key] = result
            return result

    def _recover_intent_with_separate_analysis(
        self,
        *,
        intent: IntentName,
        message: str,
        guideline: str,
    ) -> IntentDecision | None:
        normalized = str(message or "").strip()
        if not normalized or self.llm_client is None:
            return None
        if intent not in CRITICAL_INTENTS_NO_BUDGET and not self._reserve_aux_llm_slot(
            f"intent_recovery:{intent.value}"
        ):
            return None

        prompt = (
            "You are a recovery intent classifier.\n"
            "Primary JSON classifier was inconclusive, run a separate analysis pass.\n"
            "Decide whether the message explicitly matches the target intent.\n"
            "Return ONLY valid JSON with this schema:\n"
            '{"match": true, "confidence": 0.0, "reason": "short"}\n'
            f"Intent: {intent.value}\n"
            f"Guideline: {guideline}\n"
            f"User message: {normalized}\n"
        )
        try:
            response = self.llm_client.chat_completion(
                model=self.intent_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=80,
                response_format={"type": "json_object"},
            )
            logger.debug(
                "[INTENT_RAW_FULL] intent=%s full_response=%r",
                intent.value,
                getattr(response, "raw", response),
            )
            raw, payload = self._extract_response_payload(
                response=response,
                required_keys={"match", "confidence"},
            )
            if not isinstance(payload, dict):
                logger.info(
                    "[INTENT_CLASSIFY_RECOVERY_RAW] intent=%s raw_preview=%s",
                    intent.value,
                    raw[:240].replace("\n", "\\n"),
                )
                logger.info(
                    "[INTENT_CLASSIFY_RECOVERY] intent=%s status=unknown reason_code=intent_recovery_invalid_payload",
                    intent.value,
                )
                return None
            match_bool = payload.get("match")
            if not isinstance(match_bool, bool):
                logger.info(
                    "[INTENT_CLASSIFY_RECOVERY] intent=%s status=unknown reason_code=intent_recovery_invalid_payload",
                    intent.value,
                )
                return None
            confidence = self._to_confidence(payload.get("confidence"))
            if confidence < 0.55:
                logger.info(
                    "[INTENT_CLASSIFY_RECOVERY] intent=%s status=unknown reason_code=intent_recovery_low_confidence",
                    intent.value,
                )
                return None
            if match_bool:
                result = IntentDecision.match(
                    intent=intent,
                    confidence=max(confidence, 0.61),
                    reason="separate_recovery_analysis",
                    reason_code="intent_match_recovery_analysis",
                )
            else:
                result = IntentDecision.no_match(
                    intent=intent,
                    confidence=max(confidence, 0.61),
                    reason="separate_recovery_analysis",
                    reason_code="intent_no_match_recovery_analysis",
                )
            logger.info(
                "[INTENT_CLASSIFY_RECOVERY] intent=%s status=%s reason_code=%s",
                intent.value,
                result.status,
                result.reason_code,
            )
            return result
        except Exception as exc:
            logger.warning("[INTENT_CLASSIFY_RECOVERY] intent=%s failed: %s", intent.value, exc)
            return None

    def _classify_intent(self, *, intent: IntentName, message: str, fallback: bool, guideline: str) -> bool:
        """Классифицирует intent отдельным LLM-запросом с кешированием.

        Args:
            intent: Имя классифицируемого интента.
            message: Входной текст для классификации.
            fallback: Значение по умолчанию при ошибке/низкой уверенности.
            guideline: Краткая инструкция для intent-классификатора.

        Returns:
            `True`, если интент распознан уверенно, иначе `False`/`fallback`.
        """
        normalized = str(message or "").strip()
        if not normalized:
            return False
        cache_key = (intent.value, normalized)
        if cache_key in self._intent_cache:
            return self._intent_cache[cache_key]
        decision = self._classify_intent_with_status(
            intent=intent,
            message=normalized,
            guideline=guideline,
        )
        if decision.is_unknown:
            self._intent_cache[cache_key] = fallback
            return fallback
        self._intent_cache[cache_key] = decision.is_match
        return decision.is_match

    @staticmethod
    def _to_confidence(value: object) -> float:
        """Безопасно приводит confidence к диапазону [0.0, 1.0]."""
        try:
            num = float(value)
        except Exception:
            num = 0.0
        return max(0.0, min(1.0, num))

    @staticmethod
    def _extract_first_json_object(text: str) -> dict | None:
        raw = str(text or "").strip()
        if not raw:
            return None
        decoder = json.JSONDecoder()
        index = 0
        while index < len(raw):
            start = raw.find("{", index)
            if start < 0:
                break
            try:
                payload, end = decoder.raw_decode(raw, start)
            except json.JSONDecodeError:
                index = start + 1
                continue
            if isinstance(payload, dict):
                return payload
            index = max(end, start + 1)
        for candidate in MemoryManager._extract_braced_candidates(raw):
            normalized = MemoryManager._normalize_pythonish_json(candidate)
            try:
                payload = ast.literal_eval(normalized)
            except Exception:
                continue
            if isinstance(payload, dict):
                return payload
        return None

    @staticmethod
    def _extract_braced_candidates(text: str) -> list[str]:
        candidates: list[str] = []
        raw = str(text or "")
        for start, ch in enumerate(raw):
            if ch != "{":
                continue
            depth = 0
            for end in range(start, len(raw)):
                token = raw[end]
                if token == "{":
                    depth += 1
                elif token == "}":
                    depth -= 1
                    if depth == 0:
                        candidates.append(raw[start : end + 1])
                        break
        return candidates

    @staticmethod
    def _normalize_pythonish_json(candidate: str) -> str:
        normalized = re.sub(r":\s*true\b", ": True", str(candidate or ""), flags=re.IGNORECASE)
        normalized = re.sub(r":\s*false\b", ": False", normalized, flags=re.IGNORECASE)
        normalized = re.sub(r":\s*null\b", ": None", normalized, flags=re.IGNORECASE)
        return normalized

    def _extract_response_payload(
        self,
        *,
        response: object,
        required_keys: set[str],
    ) -> tuple[str, dict[str, object] | None]:
        raw_text = ""
        try:
            raw_text = str(response.choices[0].message.content or "").strip()  # type: ignore[attr-defined]
        except Exception:
            raw_text = ""

        payload = self._extract_first_json_object(raw_text) if raw_text else None
        if self._payload_has_required_keys(payload, required_keys=required_keys):
            assert isinstance(payload, dict)
            return raw_text, payload

        raw_response = getattr(response, "raw", None)
        nested_payload = self._find_payload_in_raw(raw_response, required_keys=required_keys)
        if isinstance(nested_payload, dict):
            if not raw_text:
                raw_text = json.dumps(nested_payload, ensure_ascii=False)
            return raw_text, nested_payload

        if not raw_text:
            raw_text = self._safe_dump_raw_response(raw_response)
        return raw_text, None

    def _find_payload_in_raw(
        self,
        value: object,
        *,
        required_keys: set[str],
        _visited: set[int] | None = None,
    ) -> dict[str, object] | None:
        if _visited is None:
            _visited = set()
        if value is None:
            return None
        oid = id(value)
        if oid in _visited:
            return None
        _visited.add(oid)

        if isinstance(value, dict):
            normalized = {str(k).strip().lower(): v for k, v in value.items()}
            if required_keys.issubset(set(normalized.keys())):
                return normalized
            for nested in value.values():
                found = self._find_payload_in_raw(nested, required_keys=required_keys, _visited=_visited)
                if isinstance(found, dict):
                    return found
            return None

        if isinstance(value, (list, tuple, set)):
            for item in value:
                found = self._find_payload_in_raw(item, required_keys=required_keys, _visited=_visited)
                if isinstance(found, dict):
                    return found
            return None

        for method_name in ("model_dump", "to_dict", "dict"):
            method = getattr(value, method_name, None)
            if callable(method):
                try:
                    dumped = method()
                except TypeError:
                    try:
                        dumped = method(exclude_none=True)
                    except Exception:
                        dumped = None
                except Exception:
                    dumped = None
                if dumped is not None and dumped is not value:
                    found = self._find_payload_in_raw(dumped, required_keys=required_keys, _visited=_visited)
                    if isinstance(found, dict):
                        return found
        return None

    @staticmethod
    def _payload_has_required_keys(payload: dict | None, *, required_keys: set[str]) -> bool:
        if not isinstance(payload, dict):
            return False
        keys = {str(k).strip().lower() for k in payload.keys()}
        return required_keys.issubset(keys)

    @staticmethod
    def _safe_dump_raw_response(raw_response: object) -> str:
        if raw_response is None:
            return ""
        for method_name in ("model_dump_json",):
            method = getattr(raw_response, method_name, None)
            if callable(method):
                try:
                    dumped = method()
                except Exception:
                    dumped = ""
                if dumped:
                    return str(dumped)
        for method_name in ("model_dump", "to_dict", "dict"):
            method = getattr(raw_response, method_name, None)
            if callable(method):
                try:
                    dumped_obj = method()
                except TypeError:
                    try:
                        dumped_obj = method(exclude_none=True)
                    except Exception:
                        dumped_obj = None
                except Exception:
                    dumped_obj = None
                if dumped_obj is not None:
                    try:
                        return json.dumps(dumped_obj, ensure_ascii=False)
                    except Exception:
                        return repr(dumped_obj)
        if isinstance(raw_response, (dict, list, tuple, set)):
            try:
                return json.dumps(raw_response, ensure_ascii=False)
            except Exception:
                return repr(raw_response)
        return repr(raw_response)
