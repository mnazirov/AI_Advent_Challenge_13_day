from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
import logging
import os
import re
from datetime import datetime
from time import perf_counter
from typing import Any

from context_strategies import ContextStrategyManager
from llm import OpenAILLMClient
from memory import MemoryManager, TaskContext, TaskState
from memory.response_invariants import ParsedMarkers, ResponseSignals, parse_response_markers, validate_response
from storage import ensure_session

SYSTEM_PROMPT = """You are a smart, concise, and helpful assistant on iOS. 
You are integrated into the user's daily workflow and have access 
to their device context.

## Core Behavior
- Always respond in the user's language (auto-detect)
- Keep responses SHORT — optimized for voice playback and glanceable text
- Prioritize actionable answers over lengthy explanations
- Never ask more than ONE clarifying question at a time

## Tone & Style
- Friendly but efficient — no filler phrases like "Great question!"
- Use plain language, avoid jargon unless user is technical
- For voice: use natural speech patterns, no markdown, no bullet points
- For text: use minimal formatting, emojis only when contextually appropriate

## Response Format Rules
- Simple facts → 1 sentence
- Instructions → numbered steps, max 5
- Ambiguous requests → make a reasonable assumption, state it briefly, then answer
- Unknown info → say so clearly, suggest where to find it

## Capabilities Awareness
- You CAN: set reminders, answer questions, draft messages, 
  summarize content, translate, calculate, suggest actions
- You CANNOT: access real-time data unless tools are provided, 
  make calls/purchases without explicit confirmation

## Safety & Confirmation
- Always confirm before: sending messages, making purchases, 
  deleting data, sharing personal information
- For sensitive actions say: "Just to confirm — [action]. Proceed?"

## Context Handling
- Remember context within the current session
- If user seems rushed → ultra-short replies
- If user seems exploratory → slightly more detail is okay
"""

logger = logging.getLogger("agent")


@dataclass(frozen=True)
class MarkerNormalizationEvent:
    parsed: ParsedMarkers
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class StepNormalizationContext:
    current_step_in_memory: int
    total_steps: int
    response_signals: ResponseSignals


class RollbackReason(str, Enum):
    EXPLICIT_MARKER = "explicit_marker"
    USER_PLAN_CHANGE = "user_plan_change"
    CRITICAL_INVARIANT = "critical_invariant"
    NOT_ALLOWED = "not_allowed"


@dataclass(frozen=True)
class RollbackDecision:
    should_rollback: bool
    reason: RollbackReason


class RollbackPolicy:
    """Single source of truth for rollback eligibility."""

    ALLOWED_REASONS: frozenset[RollbackReason] = frozenset(
        {
            RollbackReason.EXPLICIT_MARKER,
            RollbackReason.USER_PLAN_CHANGE,
            RollbackReason.CRITICAL_INVARIANT,
        }
    )

    @classmethod
    def is_allowed(cls, reason: RollbackReason) -> bool:
        return reason in cls.ALLOWED_REASONS


class IOSAgent:
    """iOS product assistant with memory layers and internal workflow orchestration."""

    DEFAULT_MODEL = "gpt-5-mini"
    DEFAULT_USER_ID = "default_local_user"
    MODEL_FALLBACK_ORDER = ("gpt-5-mini", "gpt-4o-mini")
    COST_PER_1M = {
        "gpt-5.2": {"input": 1.75, "output": 14.00},
        "gpt-5.1": {"input": 1.25, "output": 10.00},
        "gpt-5": {"input": 1.25, "output": 10.00},
        "gpt-5-mini": {"input": 0.25, "output": 2.00},
        "gpt-5-nano": {"input": 0.05, "output": 0.40},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4.1": {"input": 2.00, "output": 8.00},
        "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
        "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
        "o3": {"input": 2.00, "output": 8.00},
        "o3-mini": {"input": 1.10, "output": 4.40},
        "o1": {"input": 15.00, "output": 60.00},
        "o1-mini": {"input": 1.10, "output": 4.40},
        "gpt-4-turbo": {"input": 10.00, "output": 30.00},
        "gpt-4": {"input": 30.00, "output": 60.00},
        "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    }

    def __init__(self, model: str | None = None):
        self._load_env_if_needed()
        self.llm_client = OpenAILLMClient()

        requested_model = (model or os.getenv("OPENAI_MODEL") or self.DEFAULT_MODEL).strip()
        try:
            self.model = self._validate_model(requested_model)
        except ValueError:
            self.model = self.DEFAULT_MODEL
            logger.warning(
                "[INIT] Unknown model %s, fallback to default: %s",
                requested_model,
                self.model,
            )

        self.conversation_history: list[dict[str, str]] = []
        self.ctx = ContextStrategyManager(client=self.llm_client, model=self.model)
        self.memory = MemoryManager(
            short_term_limit=30,
            llm_client=self.llm_client,
            step_parser_model="gpt-5-nano",
        )

        self.last_token_stats: dict[str, Any] | None = None
        self.last_memory_stats: dict[str, Any] | None = None
        self.last_prompt_preview: dict[str, Any] | None = None
        self.last_chat_response_meta: dict[str, Any] | None = None
        logger.info("[INIT] IOSAgent initialized model=%s", self.model)

    @classmethod
    def available_models(cls) -> list[str]:
        return list(cls.COST_PER_1M.keys())

    @classmethod
    def _validate_model(cls, model: str) -> str:
        normalized = (model or "").strip()
        if not normalized:
            raise ValueError("Не указана модель")
        if normalized not in cls.COST_PER_1M:
            supported = ", ".join(cls.available_models())
            raise ValueError(f"Неподдерживаемая модель: {normalized}. Доступны: {supported}")
        return normalized

    def set_model(self, model: str) -> str:
        validated = self._validate_model(model)
        if validated == self.model:
            return self.model
        self.model = validated
        self.ctx.set_model(validated)
        logger.info("[MODEL] Модель переключена на %s", validated)
        return validated

    def chat(
        self,
        user_message: str,
        session_id: str | None = None,
        user_id: str | None = None,
        client_intent: dict[str, Any] | None = None,
    ) -> str:
        """Handles one chat turn with memory-aware prompt building."""
        t_start = perf_counter()

        current_session_id = str(session_id or "default_session")
        current_user_id = str(user_id or self.DEFAULT_USER_ID)
        ensure_session(current_session_id)
        self.memory.begin_turn_aux_budget(limit=12)
        ctx_before_turn = self.memory.working.load(current_session_id)
        state_before_turn = ctx_before_turn.state if ctx_before_turn else None

        gate_response = self._handle_planning_gate(
            session_id=current_session_id,
            user_id=current_user_id,
            user_message=user_message,
            started_at=t_start,
            client_intent=client_intent,
        )
        state_after_gate_ctx = self.memory.working.load(current_session_id)
        state_after_gate = state_after_gate_ctx.state if state_after_gate_ctx else None
        if gate_response is not None and state_before_turn == TaskState.VALIDATION and state_after_gate == TaskState.DONE:
            return gate_response
        auto_after_gate = self._auto_state_entry_response(
            session_id=current_session_id,
            user_id=current_user_id,
            user_message=user_message,
            started_at=t_start,
            prev_state=state_before_turn,
            next_state=state_after_gate,
        )
        if auto_after_gate is not None:
            return auto_after_gate
        if gate_response is not None:
            return gate_response

        if self._is_memory_recall_request(user_message):
            return self._finalize_non_llm_response(
                session_id=current_session_id,
                user_id=current_user_id,
                user_message=user_message,
                assistant_message=self._build_memory_recall_response(),
                started_at=t_start,
                finish_reason="memory_recall",
                apply_hard_constraints=False,
            )

        state_before_route = state_after_gate
        self.memory.route_user_message(
            session_id=current_session_id,
            user_id=current_user_id,
            user_message=user_message,
            client_intent=client_intent,
        )
        state_after_route_ctx = self.memory.working.load(current_session_id)
        state_after_route = state_after_route_ctx.state if state_after_route_ctx else None
        auto_after_route = self._auto_state_entry_response(
            session_id=current_session_id,
            user_id=current_user_id,
            user_message=user_message,
            started_at=t_start,
            prev_state=state_before_route,
            next_state=state_after_route,
        )
        if auto_after_route is not None:
            return auto_after_route
        gate_after_route = self._handle_planning_gate(
            session_id=current_session_id,
            user_id=current_user_id,
            user_message=user_message,
            started_at=t_start,
            client_intent=client_intent,
        )
        if gate_after_route is not None:
            return gate_after_route

        shortcut_response = self._handle_state_shortcuts(
            session_id=current_session_id,
            user_id=current_user_id,
            user_message=user_message,
            started_at=t_start,
        )
        if shortcut_response is not None:
            return shortcut_response

        after_route_response = self._handle_post_route_guidance(
            session_id=current_session_id,
            user_id=current_user_id,
            user_message=user_message,
            started_at=t_start,
        )
        if after_route_response is not None:
            return after_route_response

        if self.ctx.active == "sticky_facts":
            try:
                self.ctx.strategy.update_facts(user_message=user_message, history=self.conversation_history)
            except Exception as exc:
                logger.warning("[CTX][sticky_facts] update skipped: %s", exc)
            self._backfill_sticky_goal_from_working(session_id=current_session_id)

        messages, prompt_preview, read_meta = self.memory.build_messages(
            session_id=current_session_id,
            user_id=current_user_id,
            system_instructions=SYSTEM_PROMPT,
            data_context="",
            user_query=user_message,
        )
        self.last_prompt_preview = prompt_preview

        protocol_meta: dict[str, Any] = {}
        assistant_message = ""
        finish_reason = "stop"
        prompt_tokens = 0
        completion_tokens = 0
        attempt_messages = list(messages)

        for attempt in range(2):
            response = self._create_chat_completion(
                model=self.model,
                messages=attempt_messages,
                max_tokens=4096,
                temperature=0.7,
            )
            finish_reason = str(getattr(response.choices[0], "finish_reason", "") or "stop")
            raw_reply = response.choices[0].message.content or ""
            parsed = parse_response_markers(raw_reply)

            usage = getattr(response, "usage", None)
            prompt_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
            completion_tokens += int(getattr(usage, "completion_tokens", 0) or 0)

            candidate_text = self._sanitize_reply_text(parsed.external)
            if not candidate_text:
                candidate_text = "Не удалось сформировать ответ. Уточните задачу или добавьте контекст."
            elif finish_reason == "length":
                candidate_text = candidate_text.rstrip() + "\n\n_Ответ обрезан по длине. Можно попросить продолжить._"

            assistant_message, protocol_meta = self._finalize_external_message(
                session_id=current_session_id,
                user_id=current_user_id,
                text=candidate_text,
                internal_trace=parsed.internal,
                raw_response=raw_reply,
                source="llm",
                last_user_message=user_message,
            )
            invariant_report = protocol_meta.get("invariant_report") or {}
            if invariant_report.get("overall_status") == "ok":
                break

            can_retry = bool(invariant_report.get("can_retry", False))
            if attempt == 0 and can_retry:
                retry_prompt = self._build_invariant_retry_prompt(invariant_report)
                attempt_messages = list(messages)
                attempt_messages.append({"role": "assistant", "content": raw_reply})
                attempt_messages.append({"role": "user", "content": retry_prompt})
                logger.warning("[INVARIANT] Retry requested: %s", invariant_report.get("explanation"))
                continue

            finish_reason = "invariant_fail"
            planning_recovery = self._recover_invariant_fail_from_planning(
                session_id=current_session_id,
                user_id=current_user_id,
                user_message=user_message,
            )
            if planning_recovery is not None:
                assistant_message = str(planning_recovery.get("assistant_message") or "")
                protocol_meta = dict(planning_recovery.get("protocol_meta") or {})
                finish_reason = str(planning_recovery.get("finish_reason") or "stop")
                prompt_tokens += int(planning_recovery.get("prompt_tokens", 0) or 0)
                completion_tokens += int(planning_recovery.get("completion_tokens", 0) or 0)
                read_meta_recovered = planning_recovery.get("read_meta")
                if isinstance(read_meta_recovered, dict):
                    read_meta = read_meta_recovered
                break

            assistant_message = self._build_invariant_failure_user_message(invariant_report)
            break

        self.memory.append_turn(
            session_id=current_session_id,
            user_message=user_message,
            assistant_message=assistant_message,
        )
        self._append_history(user_message=user_message, assistant_message=assistant_message)
        total_tokens = int(prompt_tokens + completion_tokens)
        cost_usd = self._estimate_cost(prompt_tokens, completion_tokens)
        latency_ms = int(round((perf_counter() - t_start) * 1000))

        self.last_memory_stats = self.memory.stats(session_id=current_session_id, user_id=current_user_id)
        self.last_token_stats = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cost_usd": float(round(cost_usd, 6)),
            "latency_ms": latency_ms,
            "scope": "chat",
            "strategy": "memory_layers",
            "ctx_stats": self.ctx.stats(self.conversation_history),
            "memory_stats": self.last_memory_stats,
            "memory_read": read_meta,
            "prompt_preview": self.last_prompt_preview,
            "finish_reason": finish_reason,
        }
        self.last_chat_response_meta = {
            "finish_reason": finish_reason,
            "working_view": self.memory.get_working_view(session_id=current_session_id),
            "working_actions": self.memory.get_working_actions(session_id=current_session_id),
            "invariant_report": protocol_meta.get("invariant_report"),
            "internal_trace": protocol_meta.get("internal_trace", ""),
        }
        logger.info(
            "[CHAT] in=%s out=%s total=%s cost=$%.6f model=%s",
            prompt_tokens,
            completion_tokens,
            total_tokens,
            cost_usd,
            self.model,
        )
        return assistant_message

    def reset(self):
        self.conversation_history = []
        self.ctx.reset_all()
        self.last_token_stats = None
        self.last_memory_stats = None
        self.last_prompt_preview = None
        self.last_chat_response_meta = None

    def clear_session_memory(self, session_id: str) -> None:
        self.memory.clear_session(session_id=session_id)

    def restore_memory_session(self, session_id: str, messages: list[dict[str, str]] | None = None) -> None:
        if messages:
            self.memory.hydrate_short_term(session_id=session_id, messages=messages)
            return
        self.memory.short_term.get_context(session_id)

    def _handle_planning_gate(
        self,
        *,
        session_id: str,
        user_id: str,
        user_message: str,
        started_at: float,
        client_intent: dict[str, Any] | None = None,
    ) -> str | None:
        guard_ctx_before = self.memory.working.load(session_id)
        gate_message = self.memory.enforce_planning_gate(
            session_id=session_id,
            user_message=user_message,
            user_id=user_id,
            client_intent=client_intent,
        )
        if gate_message == self.memory.VALIDATION_CONFIRMED_SIGNAL:
            return self._finalize_done_transition(
                session_id=session_id,
                user_id=user_id,
                user_message=user_message,
                started_at=started_at,
            )
        if gate_message is None:
            return None

        is_planning_block = bool(guard_ctx_before and guard_ctx_before.state.value == "PLANNING")
        finish_reason = "state_blocked_planning" if is_planning_block else "state_blocked"
        if is_planning_block:
            assistant_message = self._sanitize_reply_text(gate_message)
        else:
            if guard_ctx_before and guard_ctx_before.state == TaskState.VALIDATION:
                assistant_message = self._sanitize_reply_text(gate_message)
                return self._finalize_non_llm_response(
                    session_id=session_id,
                    user_id=user_id,
                    user_message=user_message,
                    assistant_message=assistant_message,
                    started_at=started_at,
                    finish_reason=finish_reason,
                )
            working_view = self.memory.get_working_view(session_id=session_id)
            step_title = str(working_view.get("current_step") or "").strip() or "текущий шаг"
            total_steps = int(working_view.get("total_steps") or 0)
            step_index_raw = working_view.get("step_index")
            step_index = int(step_index_raw) if isinstance(step_index_raw, (int, float)) else None
            gate_lower = gate_message.lower()
            if (
                "финальн" in gate_lower
                or "все шаги выполнены" in gate_lower
                or "пропуск проверк" in gate_lower
                or "ответьте yes или no" in gate_lower
                or "уже завершена" in gate_lower
            ):
                assistant_message = self._sanitize_reply_text(gate_message)
                return self._finalize_non_llm_response(
                    session_id=session_id,
                    user_id=user_id,
                    user_message=user_message,
                    assistant_message=assistant_message,
                    started_at=started_at,
                    finish_reason=finish_reason,
                )
            if step_index is None and total_steps > 0:
                done_steps = working_view.get("done") if isinstance(working_view, dict) else []
                done_count = len(done_steps) if isinstance(done_steps, list) else 0
                step_index = min(max(done_count + 1, 1), total_steps)
            if step_index is not None and total_steps > 0:
                assistant_message = self._sanitize_reply_text(
                    f"Сейчас выполняется шаг {step_index}/{total_steps}: «{step_title}».\n"
                    "Завершите текущий шаг, и я автоматически переведу задачу дальше."
                )
            else:
                assistant_message = self._sanitize_reply_text(gate_message)
        return self._finalize_non_llm_response(
            session_id=session_id,
            user_id=user_id,
            user_message=user_message,
            assistant_message=assistant_message,
            started_at=started_at,
            finish_reason=finish_reason,
        )

    def _auto_state_entry_response(
        self,
        *,
        session_id: str,
        user_id: str,
        user_message: str,
        started_at: float,
        prev_state: TaskState | None,
        next_state: TaskState | None,
    ) -> str | None:
        if not next_state or prev_state == next_state:
            return None
        ctx = self.memory.working.load(session_id)
        if not ctx:
            return None

        if next_state == TaskState.VALIDATION:
            return self._finalize_non_llm_response(
                session_id=session_id,
                user_id=user_id,
                user_message=user_message,
                assistant_message=self.build_validation_prompt(ctx),
                started_at=started_at,
                finish_reason="state_auto_validation",
            )

        if next_state == TaskState.DONE:
            summary = self.build_done_summary_prompt(ctx)
            try:
                self.memory.save_done_summary_to_long_term(
                    session_id=session_id,
                    user_id=user_id,
                    task_title=ctx.task,
                    summary=summary,
                )
            except Exception as exc:
                logger.warning("[DONE_SUMMARY_SAVE] skipped: %s", exc)
            return self._finalize_non_llm_response(
                session_id=session_id,
                user_id=user_id,
                user_message=user_message,
                assistant_message=summary,
                started_at=started_at,
                finish_reason="state_auto_done",
            )
        return None

    def _finalize_done_transition(
        self,
        *,
        session_id: str,
        user_id: str,
        user_message: str,
        started_at: float,
    ) -> str:
        ctx = self.memory.working.load(session_id)
        if not ctx:
            return self._finalize_non_llm_response(
                session_id=session_id,
                user_id=user_id,
                user_message=user_message,
                assistant_message="Не удалось завершить задачу: рабочий контекст не найден.",
                started_at=started_at,
                finish_reason="state_done_error",
            )

        # Build and persist summary before freezing working memory in DONE.
        summary = self.build_done_summary_prompt(ctx)
        try:
            self.memory.save_done_summary_to_long_term(
                session_id=session_id,
                user_id=user_id,
                task_title=ctx.task,
                summary=summary,
            )
        except Exception as exc:
            logger.warning("[DONE_SUMMARY_SAVE] skipped: %s", exc)

        try:
            self.memory.working.transition_state(ctx, TaskState.DONE)
            ctx.updated_at = datetime.utcnow().isoformat()
            self.memory.working.save(ctx)
        except ValueError as exc:
            return self._finalize_non_llm_response(
                session_id=session_id,
                user_id=user_id,
                user_message=user_message,
                assistant_message=f"Не удалось завершить задачу: {exc}",
                started_at=started_at,
                finish_reason="state_done_error",
            )

        return self._finalize_non_llm_response(
            session_id=session_id,
            user_id=user_id,
            user_message=user_message,
            assistant_message=summary,
            started_at=started_at,
            finish_reason="state_auto_done",
        )

    def build_validation_prompt(self, task: TaskContext) -> str:
        steps_summary = "\n".join(f"✓ {step}" for step in (task.done or []))
        if not steps_summary:
            steps_summary = "✓ Нет завершённых шагов"
        return (
            f"Задача «{task.task}» завершила все шаги.\n\n"
            "Итог выполненных шагов:\n"
            f"{steps_summary}\n\n"
            "Перед тем как зафиксировать результат:\n"
            "1. Всё ли работает как ожидалось?\n"
            "2. Есть ли замечания или что-то требует доработки?\n\n"
            "Если есть замечания — опишите их сообщением, вернёмся к доработке.\n"
            "Если замечаний нет — продолжим автоматически."
        )

    def build_done_summary_prompt(self, task: TaskContext) -> str:
        steps = "\n".join(f"✓ {step}" for step in (task.done or []))
        if not steps:
            steps = "✓ Нет зафиксированных шагов"

        artifact_lines: list[str] = []
        for artifact in task.artifacts or []:
            art_type = getattr(getattr(artifact, "type", None), "value", "")
            art_ref = str(getattr(artifact, "ref", "") or "").strip()
            if art_ref:
                label = art_type or "artifact"
                artifact_lines.append(f"- {label}: {art_ref}")
        artifacts = "\n".join(artifact_lines) if artifact_lines else "нет сохранённых артефактов"

        decisions_raw = (task.vars or {}).get("requirements") if isinstance(task.vars, dict) else []
        decisions: list[str] = [str(item).strip() for item in (decisions_raw or []) if str(item).strip()]
        decisions_block = "\n".join(f"- {item}" for item in decisions[:5])
        if not decisions_block:
            decisions_block = (
                "Сохрани ключевые решения, паттерны и договорённости "
                "которые были приняты в ходе этой задачи в долгосрочную память."
            )

        return (
            f"## Задача завершена: «{task.task}»\n\n"
            "### Выполненные шаги\n"
            f"{steps}\n\n"
            "### Артефакты\n"
            f"{artifacts}\n\n"
            "### Архитектурные решения\n"
            f"{decisions_block}\n\n"
            "Что дальше?\n"
            "- Начать следующую задачу в рамках проекта\n"
            "- Или задай любой вопрос по итогам"
        )

    def _handle_post_route_guidance(
        self,
        *,
        session_id: str,
        user_id: str,
        user_message: str,
        started_at: float,
    ) -> str | None:
        guard_ctx = self.memory.working.load(session_id)
        if not (
            guard_ctx
            and guard_ctx.state.value == "PLANNING"
            and bool((guard_ctx.vars or {}).get("plan_guidance_required"))
        ):
            return None

        goal = str(guard_ctx.task or "Текущая задача").strip() or "Текущая задача"
        assistant_message = self._sanitize_reply_text(
            f"Задача создана: '{goal}'.\n"
            "Чтобы начать выполнение, добавьте шаги плана.\n"
            "Можно описать шаги вручную или запросить автоплан."
        )
        vars_patch = dict(guard_ctx.vars)
        vars_patch.pop("plan_guidance_required", None)
        self.memory.working.update(session_id, vars=vars_patch)

        return self._finalize_non_llm_response(
            session_id=session_id,
            user_id=user_id,
            user_message=user_message,
            assistant_message=assistant_message,
            started_at=started_at,
            finish_reason="state_blocked_planning",
        )

    def _backfill_sticky_goal_from_working(self, *, session_id: str) -> None:
        if self.ctx.active != "sticky_facts":
            return
        strategy = self.ctx.strategy
        facts = getattr(strategy, "facts", None)
        if not isinstance(facts, dict):
            return
        if str(facts.get("goal") or "").strip():
            return

        extract_meta = self.memory.router.get_last_working_extract_meta(session_id)
        extract_applied = bool(extract_meta.get("applied"))
        try:
            extract_confidence = float(extract_meta.get("confidence", 0.0) or 0.0)
        except Exception:
            extract_confidence = 0.0
        if extract_applied or extract_confidence >= 0.7:
            return

        working_ctx = self.memory.working.load(session_id)
        if not working_ctx:
            return
        task_goal = str(working_ctx.task or "").strip()
        if not task_goal:
            return
        facts["goal"] = task_goal
        logger.info("[WORKING_EXTRACT_FALLBACK] goal взят из working.task: %s", task_goal)

    def _handle_state_shortcuts(
        self,
        *,
        session_id: str,
        user_id: str,
        user_message: str,
        started_at: float,
    ) -> str | None:
        ctx = self.memory.working.load(session_id)
        if not ctx:
            return None

        msg = str(user_message or "").strip()
        lower = msg.lower()

        if ctx.state == TaskState.EXECUTION:
            if "@stateobject" in lower and "@observedobject" in lower:
                step = str(ctx.current_step or "текущий шаг")
                assistant_message = (
                    f"В контексте шага «{step}» используем `@StateObject`, потому что View создаёт и владеет "
                    "ViewModel. У `@StateObject` корректный lifetime для ownership внутри этого экрана, "
                    "а `@ObservedObject` берём только когда объект уже создан снаружи и просто пробрасывается в View."
                )
                return self._finalize_non_llm_response(
                    session_id=session_id,
                    user_id=user_id,
                    user_message=user_message,
                    assistant_message=assistant_message,
                    started_at=started_at,
                    finish_reason="state_execution_context",
                )

            if self._is_code_request(msg):
                step = str(ctx.current_step or "Текущий шаг")
                assistant_message = (
                    f"Шаг: {step}\n\n"
                    "```swift\n"
                    "import SwiftUI\n\n"
                    "@MainActor\n"
                    "final class LoginViewModel: ObservableObject {\n"
                    "    @Published var email = \"\"\n"
                    "    @Published var password = \"\"\n"
                    "    @Published var errorText: String?\n\n"
                    "    var isValid: Bool {\n"
                    "        let emailOk = email.contains(\"@\") && email.contains(\".\")\n"
                    "        let passOk = password.count >= 8\n"
                    "        return emailOk && passOk\n"
                    "    }\n\n"
                    "    func submit() {\n"
                    "        guard isValid else {\n"
                    "            errorText = \"Проверь email и пароль\"\n"
                    "            return\n"
                    "        }\n"
                    "        errorText = nil\n"
                    "    }\n"
                    "}\n\n"
                    "struct LoginView: View {\n"
                    "    @StateObject private var vm = LoginViewModel()\n\n"
                    "    var body: some View {\n"
                    "        VStack(spacing: 12) {\n"
                    "            TextField(\"Email\", text: $vm.email)\n"
                    "                .keyboardType(.emailAddress)\n"
                    "                .textInputAutocapitalization(.never)\n"
                    "                .autocorrectionDisabled()\n"
                    "            SecureField(\"Пароль\", text: $vm.password)\n"
                    "            if let error = vm.errorText {\n"
                    "                Text(error).foregroundColor(.red)\n"
                    "            }\n"
                    "            Button(\"Войти\") { vm.submit() }\n"
                    "                .disabled(!vm.isValid)\n"
                    "        }\n"
                    "        .padding()\n"
                    "    }\n"
                    "}\n"
                    "```"
                )
                return self._finalize_non_llm_response(
                    session_id=session_id,
                    user_id=user_id,
                    user_message=user_message,
                    assistant_message=assistant_message,
                    started_at=started_at,
                    finish_reason="state_execution_code",
                )

        if ctx.state == TaskState.VALIDATION and self.memory.is_validation_checklist_request(msg):
            assistant_message = (
                "Финальный чеклист перед отправкой:\n"
                "1. Поля email и пароль валидируются до отправки.\n"
                "2. Ошибки validation видны рядом с полями.\n"
                "3. Кнопка входа неактивна при невалидных данных.\n"
                "4. Для авторизации обработаны success/error сценарии.\n"
                "5. Текст ошибок и состояние loading не ломают layout."
            )
            return self._finalize_non_llm_response(
                session_id=session_id,
                user_id=user_id,
                user_message=user_message,
                assistant_message=assistant_message,
                started_at=started_at,
                finish_reason="state_validation_checklist",
            )

        if ctx.state == TaskState.DONE:
            if self._is_save_memory_request(msg):
                summary = self.build_done_summary_prompt(ctx)
                try:
                    self.memory.save_done_summary_to_long_term(
                        session_id=session_id,
                        user_id=user_id,
                        task_title=ctx.task,
                        summary=summary,
                    )
                    assistant_message = "Сохранил ключевые решения и итог задачи в долгосрочную память."
                    finish_reason = "state_done_saved"
                except Exception as exc:
                    assistant_message = f"Не удалось сохранить в память: {exc}"
                    finish_reason = "state_done_save_error"
                return self._finalize_non_llm_response(
                    session_id=session_id,
                    user_id=user_id,
                    user_message=user_message,
                    assistant_message=assistant_message,
                    started_at=started_at,
                    finish_reason=finish_reason,
                )

            if self._is_next_steps_request(msg):
                assistant_message = (
                    "Рекомендованные следующие шаги:\n"
                    "1. Вынести хранение token в Keychain.\n"
                    "2. Добавить навигацию после login (Coordinator/Router).\n"
                    "3. Подключить biometric вход как опцию.\n"
                    "4. Написать UI и интеграционные тесты на auth flow."
                )
                return self._finalize_non_llm_response(
                    session_id=session_id,
                    user_id=user_id,
                    user_message=user_message,
                    assistant_message=assistant_message,
                    started_at=started_at,
                    finish_reason="state_done_next_steps",
                )

        return None

    @staticmethod
    def _is_code_request(message: str) -> bool:
        lower = str(message or "").lower()
        markers = [
            "покажи код",
            "дай код",
            "код для текущего шага",
            "пример кода",
        ]
        return any(marker in lower for marker in markers)

    @staticmethod
    def _is_next_steps_request(message: str) -> bool:
        lower = str(message or "").lower()
        markers = ["что дальше", "дальше рекоменду", "следующие шаги", "what next"]
        return any(marker in lower for marker in markers)

    @staticmethod
    def _is_save_memory_request(message: str) -> bool:
        lower = str(message or "").lower()
        return "сохрани" in lower and ("памят" in lower or "ключев" in lower)

    def _finalize_non_llm_response(
        self,
        *,
        session_id: str,
        user_id: str,
        user_message: str,
        assistant_message: str,
        started_at: float,
        finish_reason: str,
        apply_hard_constraints: bool = True,
    ) -> str:
        formatted_message, protocol_meta = self._finalize_external_message(
            session_id=session_id,
            user_id=user_id,
            text=assistant_message,
            internal_trace="",
            apply_hard_constraints=apply_hard_constraints,
            source="non_llm",
            last_user_message=user_message,
        )
        self.memory.append_turn(
            session_id=session_id,
            user_message=user_message,
            assistant_message=formatted_message,
        )
        self._append_history(user_message=user_message, assistant_message=formatted_message)

        latency_ms = int(round((perf_counter() - started_at) * 1000))
        self.last_memory_stats = self.memory.stats(session_id=session_id, user_id=user_id)
        self.last_token_stats = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cost_usd": 0.0,
            "latency_ms": latency_ms,
            "scope": "chat",
            "strategy": "memory_layers",
            "ctx_stats": self.ctx.stats(self.conversation_history),
            "memory_stats": self.last_memory_stats,
            "finish_reason": finish_reason,
        }
        self.last_prompt_preview = {
            "schema_version": 2,
            "system": "[REDACTED_SYSTEM_PROMPT]",
            "system_chars": 0,
            "system_hash": "",
            "user_chars": len(user_message or ""),
            "short_term_count": 0,
            "working_state": self.memory.get_working_view(session_id=session_id).get("state"),
            "profile_injected": [],
            "profile_skipped": [],
            "decisions_count": 0,
            "notes_count": 0,
        }
        self.last_chat_response_meta = {
            "finish_reason": finish_reason,
            "working_view": self.memory.get_working_view(session_id=session_id),
            "working_actions": self.memory.get_working_actions(session_id=session_id),
            "invariant_report": protocol_meta.get("invariant_report"),
            "internal_trace": protocol_meta.get("internal_trace", ""),
        }
        return formatted_message

    def _finalize_external_message(
        self,
        *,
        session_id: str,
        user_id: str,
        text: str,
        internal_trace: str,
        apply_hard_constraints: bool = True,
        raw_response: str | None = None,
        source: str = "non_llm",
        last_user_message: str = "",
    ) -> tuple[str, dict[str, Any]]:
        payload_raw = raw_response if raw_response is not None else self._build_marker_payload(text=text, session_id=session_id)
        parsed = parse_response_markers(payload_raw)

        marker_events: list[dict[str, Any]] = []
        raw_marker_event = self._apply_parsed_markers_to_working(
            session_id=session_id,
            parsed=parsed,
            event_source="raw",
            last_user_message=last_user_message,
        )
        if raw_marker_event:
            marker_events.append(raw_marker_event)
        guard_event = self._run_execution_guard(
            session_id=session_id,
            response_signals=parsed.to_response_signals(),
        )
        if guard_event:
            marker_events.append(guard_event)

        cleaned = self._strip_internal_artifacts(self._sanitize_reply_text(parsed.external or text))
        if apply_hard_constraints:
            cleaned = self._enforce_hard_constraints_on_reply(user_id=user_id, reply=cleaned)
        if not cleaned:
            cleaned = "Готово, продолжаем."

        normalized = cleaned.rstrip()

        validation_report, parsed_for_validation, normalization_event = self._run_post_validation(
            session_id=session_id,
            parsed=parsed,
            source=source,
            fallback_text=normalized,
        )
        if normalization_event is not None:
            normalized_event = self._apply_parsed_markers_to_working(
                session_id=session_id,
                parsed=normalization_event.parsed,
                event_source="normalization",
                last_user_message=last_user_message,
            )
            if normalized_event:
                marker_events.append(normalized_event)
            guard_after_normalization = self._run_execution_guard(
                session_id=session_id,
                response_signals=normalization_event.parsed.to_response_signals(),
            )
            if guard_after_normalization:
                marker_events.append(guard_after_normalization)
        if validation_report.get("overall_status") != "ok":
            normalized = self._build_invariant_failure_user_message(validation_report)

        meta = {
            "invariant_report": validation_report,
            "internal_trace": parsed.internal or internal_trace,
            "parsed_markers": parsed_for_validation.to_dict(),
            "parsed_markers_raw": parsed.to_dict(),
            "marker_events": marker_events,
        }
        return normalized.strip(), meta

    def _split_internal_external(self, text: str) -> tuple[str, str]:
        parsed = parse_response_markers(str(text or ""))
        if parsed.external:
            return parsed.internal, parsed.external
        raw = str(text or "")
        if not raw.strip():
            return "", ""
        try:
            parsed_json = json.loads(raw)
            if isinstance(parsed_json, dict):
                return (
                    str(parsed_json.get("internal") or "").strip(),
                    str(parsed_json.get("external") or "").strip(),
                )
        except Exception:
            pass
        return parsed.internal, raw.strip()

    def _strip_internal_artifacts(self, text: str) -> str:
        source = str(text or "").strip()
        if not source:
            return ""
        source = re.sub(r"<internal>.*?</internal>", "", source, flags=re.IGNORECASE | re.DOTALL)
        source = re.sub(r"</?external>", "", source, flags=re.IGNORECASE)
        lines = [line.rstrip() for line in source.splitlines()]
        cleaned_lines: list[str] = []
        for line in lines:
            lower = line.strip().lower()
            if re.match(r"^>\s*\[v\d+\.\d+", line.strip(), flags=re.IGNORECASE):
                continue
            if lower.startswith("inv:") or lower.startswith("next:"):
                continue
            if re.match(r"^\[step_done:\s*\d+\]\s*$", line.strip(), flags=re.IGNORECASE):
                continue
            if re.match(r"^\[next_state:\s*[a-z_]+\]\s*$", line.strip(), flags=re.IGNORECASE):
                continue
            if re.match(r"^\[validation_ok\]\s*$", line.strip(), flags=re.IGNORECASE):
                continue
            if re.match(r"^\[validation_fail:\s*.*\]\s*$", line.strip(), flags=re.IGNORECASE):
                continue
            if re.match(r"^\[done:\s*.*\]\s*$", line.strip(), flags=re.IGNORECASE):
                continue
            if re.match(r"^\[open_question:\s*.*\]\s*$", line.strip(), flags=re.IGNORECASE):
                continue
            if re.match(r"^\[code_artifact(?:\s*:\s*.*)?\]\s*$", line.strip(), flags=re.IGNORECASE):
                continue
            cleaned_lines.append(line)
        return "\n".join(cleaned_lines).strip()

    def _build_state_object_payload(self, *, session_id: str) -> dict[str, Any]:
        ctx = self.memory.working.load(session_id)
        if ctx:
            return {
                "task": str(ctx.task or ""),
                "state": str(ctx.state.value),
                "plan": list(ctx.plan),
                "current_step": ctx.current_step,
                "done": list(ctx.done),
                "artifacts": [a.to_dict() for a in (ctx.artifacts or [])],
                "open_questions": list(ctx.open_questions),
            }
        return {
            "task": "",
            "state": "PLANNING",
            "plan": [],
            "current_step": None,
            "done": [],
            "artifacts": [],
            "open_questions": [],
        }

    def _build_marker_payload(self, *, text: str, session_id: str) -> str:
        state_payload = self._build_state_object_payload(session_id=session_id)
        done_count = len(state_payload.get("done") or [])
        state_value = str(state_payload.get("state") or "PLANNING").strip().upper() or "PLANNING"
        return (
            "<internal>\n"
            f"[STEP_DONE: {done_count}]\n"
            f"[NEXT_STATE: {state_value}]\n"
            "[VALIDATION_OK]\n"
            "</internal>\n"
            "<external>\n"
            f"{str(text or '').strip()}\n"
            "</external>"
        )

    def _run_post_validation(
        self,
        *,
        session_id: str,
        parsed: ParsedMarkers,
        source: str,
        fallback_text: str,
    ) -> tuple[dict[str, Any], ParsedMarkers, MarkerNormalizationEvent | None]:
        try:
            state_payload = self._build_state_object_payload(session_id=session_id)
            parsed_for_validation = parsed
            normalization_event: MarkerNormalizationEvent | None = None
            marker_reasons: list[str] = []
            expected_done = len(state_payload.get("done") or [])
            marker_shape_invalid = (
                parsed.step_done is None
                or not bool(parsed.next_state_marker_present)
                or not parsed.has_validation_marker()
                or (parsed.step_done is not None and parsed.step_done != expected_done)
            )
            if source == "llm" and marker_shape_invalid:
                parsed_for_validation, marker_reasons = self._normalize_markers_for_validation(
                    parsed=parsed,
                    state_payload=state_payload,
                    fallback_text=fallback_text,
                )
                normalization_event = MarkerNormalizationEvent(
                    parsed=parsed_for_validation,
                    reasons=tuple(marker_reasons),
                )
                logger.info(
                    "[INVARIANT] marker normalization applied source=%s reasons=%s",
                    source,
                    ",".join(marker_reasons) if marker_reasons else "none",
                )

            allowed_transitions = {
                str(src.value): {str(dst.value) for dst in dst_set}
                for src, dst_set in self.memory.working.ALLOWED_TRANSITIONS.items()
            }
            report = validate_response(
                parsed=parsed_for_validation,
                state_object=state_payload,
                allowed_transitions=allowed_transitions,
            ).to_dict()
            if normalization_event is not None:
                report["normalization"] = list(normalization_event.reasons)
            if report.get("overall_status") != "ok":
                report["can_retry"] = bool(source == "llm")
            return report, parsed_for_validation, normalization_event
        except Exception as exc:
            logger.exception("[INVARIANT] post-validation crashed: %s", exc)
            report = {
                "overall_status": "fail",
                "violations": [{"code": "VALIDATOR_EXCEPTION", "message": str(exc)}],
                "can_retry": False,
                "explanation": f"Не удалось проверить инварианты: {exc}",
            }
            return report, parsed, None

    def _normalize_markers_for_validation(
        self,
        *,
        parsed: ParsedMarkers,
        state_payload: dict[str, Any],
        fallback_text: str,
    ) -> tuple[ParsedMarkers, list[str]]:
        reasons: list[str] = []
        done_in_memory = len(state_payload.get("done") or [])
        total_steps = len(state_payload.get("plan") or [])
        response_signals = parsed.to_response_signals()
        inferred_step_done = self._infer_completed_step(
            StepNormalizationContext(
                current_step_in_memory=done_in_memory,
                total_steps=total_steps,
                response_signals=response_signals,
            )
        )

        step_done_value = parsed.step_done
        if step_done_value is None:
            reasons.append("missing_step_done")
            step_done_value = inferred_step_done
            logger.info(
                "[STEP_NORMALIZE] inferred step_done=%s (memory=%s total=%s code=%s done=%s open_q=%s)",
                step_done_value,
                done_in_memory,
                total_steps,
                response_signals.has_code_artifact,
                response_signals.has_done_phrase,
                response_signals.has_open_question,
            )
        elif step_done_value != done_in_memory:
            reasons.append("step_done_mismatch")
            step_done_value = max(done_in_memory, inferred_step_done)
            logger.info(
                "[STEP_NORMALIZE] realigned step_done=%s (raw=%s memory=%s total=%s)",
                step_done_value,
                parsed.step_done,
                done_in_memory,
                total_steps,
            )

        current_state = self._parse_task_state(state_payload.get("state"))
        normalized_state = self._parse_task_state(parsed.next_state)
        if normalized_state is None:
            reasons.append("missing_or_invalid_next_state")
            normalized_state = current_state or TaskState.PLANNING

        validation_marker = "[VALIDATION_OK]"
        if parsed.validation_fail_reason:
            validation_marker = f"[VALIDATION_FAIL: {parsed.validation_fail_reason}]"
        elif not parsed.validation_ok:
            reasons.append("missing_validation_marker")

        done_marker = f"[DONE: {parsed.done_note}]" if parsed.done_note else ""
        open_question_marker = f"[OPEN_QUESTION: {parsed.open_question_note}]" if parsed.open_question_note else ""
        code_artifact_marker = (
            f"[CODE_ARTIFACT: {parsed.code_artifact_note}]"
            if parsed.code_artifact_note and parsed.code_artifact_note != "fenced_code"
            else ("[CODE_ARTIFACT]" if parsed.code_artifact_note else "")
        )

        payload = (
            "<internal>\n"
            f"[STEP_DONE: {int(step_done_value)}]\n"
            f"[NEXT_STATE: {normalized_state.value}]\n"
            f"{validation_marker}\n"
            f"{done_marker}\n"
            f"{open_question_marker}\n"
            f"{code_artifact_marker}\n"
            "</internal>\n"
            "<external>\n"
            f"{str(fallback_text or '').strip()}\n"
            "</external>"
        )
        return parse_response_markers(payload), reasons

    @staticmethod
    def _infer_completed_step(ctx: StepNormalizationContext) -> int:
        current = max(0, int(ctx.current_step_in_memory))
        total = max(0, int(ctx.total_steps))
        signals = ctx.response_signals

        if total <= 0:
            return current
        if signals.has_code_artifact and signals.has_done_phrase and not signals.has_open_question:
            return total
        if signals.has_code_artifact:
            return min(total, current + 1)
        return min(total, current)

    @staticmethod
    def _parse_task_state(value: object) -> TaskState | None:
        raw = str(value or "").strip().upper()
        if not raw:
            return None
        try:
            return TaskState(raw)
        except ValueError:
            return None

    def _apply_parsed_markers_to_working(
        self,
        *,
        session_id: str,
        parsed: ParsedMarkers,
        event_source: str,
        last_user_message: str,
    ) -> dict[str, Any] | None:
        ctx = self.memory.working.load(session_id)
        if ctx is None:
            return None

        event: dict[str, Any] = {"source": event_source, "step_applied": False, "state_applied": False}

        if parsed.step_done is not None:
            plan_len = len(ctx.plan)
            target_done = max(0, min(int(parsed.step_done), plan_len))
            before_done = len(ctx.done)
            while True:
                fresh_ctx = self.memory.working.load(session_id)
                if fresh_ctx is None:
                    break
                if len(fresh_ctx.done) >= target_done:
                    break
                if fresh_ctx.state != TaskState.EXECUTION:
                    logger.warning(
                        "[STEP_BLOCKED] marker step_done=%s source=%s state=%s",
                        target_done,
                        event_source,
                        fresh_ctx.state.value,
                    )
                    break
                try:
                    self.memory.working.complete_current_step(session_id=session_id)
                except ValueError as exc:
                    logger.warning(
                        "[STEP_BLOCKED] marker step_done=%s source=%s reason=%s",
                        target_done,
                        event_source,
                        exc,
                    )
                    break

            after_ctx = self.memory.working.load(session_id)
            after_done = len(after_ctx.done) if after_ctx else before_done
            if after_done > before_done:
                event["step_applied"] = True
                event["step_target"] = target_done
                event["step_done"] = after_done
                logger.info(
                    "[STEP_UPDATE] current_step=%s applied after %s",
                    after_done,
                    event_source,
                )
                logger.info(
                    "[STEP_WRITTEN] current_step=%s сохранён в working layer",
                    after_done,
                )

        next_state = self._parse_task_state(parsed.next_state)
        if next_state is not None:
            state_ctx = self.memory.working.load(session_id)
            if state_ctx is not None and state_ctx.state != next_state:
                rollback_decision = self._decide_execution_rollback(
                    current_state=state_ctx.state,
                    parsed=parsed,
                    event_source=event_source,
                    last_user_message=last_user_message,
                    user_plan_change=False,
                    critical_violation=False,
                )
                if state_ctx.state == TaskState.EXECUTION and next_state == TaskState.PLANNING:
                    if not rollback_decision.should_rollback:
                        logger.info(
                            "[ROLLBACK_SKIP] %s -> %s source=%s reason=%s",
                            state_ctx.state.value,
                            next_state.value,
                            event_source,
                            rollback_decision.reason.value,
                        )
                        next_state = None
                    else:
                        logger.info(
                            "[ROLLBACK_DECISION] %s -> %s source=%s reason=%s",
                            state_ctx.state.value,
                            next_state.value,
                            event_source,
                            rollback_decision.reason.value,
                        )
                if next_state is None:
                    allowed = False
                else:
                    allowed = next_state in self.memory.working.ALLOWED_TRANSITIONS.get(state_ctx.state, set())
                if not allowed:
                    if next_state is not None:
                        logger.warning(
                            "[STATE_BLOCKED] нормализованный переход %s -> %s запрещён",
                            state_ctx.state.value,
                            next_state.value,
                        )
                else:
                    old_state = state_ctx.state
                    try:
                        self.memory.working.transition_state(state_ctx, next_state)
                        state_ctx.updated_at = datetime.utcnow().isoformat()
                        self.memory.working.save(state_ctx)
                        event["state_applied"] = True
                        event["state_from"] = old_state.value
                        event["state_to"] = next_state.value
                        logger.info(
                            "[STATE_UPDATE] %s -> %s applied after %s",
                            old_state.value,
                            next_state.value,
                            event_source,
                        )
                    except ValueError as exc:
                        logger.warning(
                            "[STATE_BLOCKED] нормализованный переход %s -> %s запрещён (%s)",
                            old_state.value,
                            next_state.value,
                            exc,
                        )

        if event.get("step_applied") or event.get("state_applied"):
            return event
        return None

    def _decide_execution_rollback(
        self,
        *,
        current_state: TaskState,
        parsed: ParsedMarkers,
        event_source: str,
        last_user_message: str,
        user_plan_change: bool,
        critical_violation: bool,
    ) -> RollbackDecision:
        if current_state != TaskState.EXECUTION:
            return RollbackDecision(should_rollback=False, reason=RollbackReason.NOT_ALLOWED)

        target_state = self._parse_task_state(parsed.next_state)
        if target_state == TaskState.PLANNING and event_source == "raw":
            return self._validate_rollback_marker(
                marker_next_state=target_state,
                last_user_message=last_user_message,
                current_state=current_state,
            )

        if user_plan_change:
            reason = RollbackReason.USER_PLAN_CHANGE
            return RollbackDecision(should_rollback=RollbackPolicy.is_allowed(reason), reason=reason)

        if critical_violation:
            reason = RollbackReason.CRITICAL_INVARIANT
            return RollbackDecision(should_rollback=RollbackPolicy.is_allowed(reason), reason=reason)

        return RollbackDecision(should_rollback=False, reason=RollbackReason.NOT_ALLOWED)

    def _validate_rollback_marker(
        self,
        *,
        marker_next_state: TaskState,
        last_user_message: str,
        current_state: TaskState,
    ) -> RollbackDecision:
        if marker_next_state != TaskState.PLANNING:
            return RollbackDecision(should_rollback=False, reason=RollbackReason.NOT_ALLOWED)

        if current_state != TaskState.EXECUTION:
            reason = RollbackReason.EXPLICIT_MARKER
            return RollbackDecision(should_rollback=RollbackPolicy.is_allowed(reason), reason=reason)

        normalized_message = str(last_user_message or "").strip()
        if not normalized_message:
            reason = RollbackReason.EXPLICIT_MARKER
            return RollbackDecision(should_rollback=RollbackPolicy.is_allowed(reason), reason=reason)

        llm_client = getattr(self, "llm_client", None)
        if llm_client is None:
            logger.info(
                "[ROLLBACK_SUPPRESSED] explicit_marker отклонён: llm unavailable (msg=%r)",
                normalized_message,
            )
            return RollbackDecision(should_rollback=False, reason=RollbackReason.NOT_ALLOWED)

        prompt = (
            f'Пользователь написал: "{normalized_message}"\n\n'
            "Пользователь просит пересмотреть весь план работы?\n"
            "Ответь ТОЛЬКО: YES или NO.\n"
            "YES — только если пользователь явно хочет изменить план целиком или начать заново.\n"
            "NO  — если пользователь отказывается от предложенной опции, уточняет деталь,\n"
            '      говорит "нет" на вопрос агента, или просто соглашается продолжить без изменений.'
        )

        try:
            rollback_model = str(
                getattr(self.memory, "intent_model", getattr(self, "model", self.DEFAULT_MODEL))
                or getattr(self, "model", self.DEFAULT_MODEL)
            )
            response = llm_client.chat_completion(
                model=rollback_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=5,
                temperature=0,
            )
            raw = str(response.choices[0].message.content or "").strip()
            if self._parse_yes_no(raw) == "YES":
                reason = RollbackReason.EXPLICIT_MARKER
                return RollbackDecision(should_rollback=RollbackPolicy.is_allowed(reason), reason=reason)
            logger.info(
                "[ROLLBACK_SUPPRESSED] explicit_marker отклонён: пользователь не менял план (msg=%r)",
                normalized_message,
            )
            return RollbackDecision(should_rollback=False, reason=RollbackReason.NOT_ALLOWED)
        except Exception as exc:
            logger.warning(
                "[ROLLBACK_SUPPRESSED] explicit_marker check failed (%s), rollback skipped (msg=%r)",
                exc,
                normalized_message,
            )
            return RollbackDecision(should_rollback=False, reason=RollbackReason.NOT_ALLOWED)

    @staticmethod
    def _parse_yes_no(raw: str) -> str:
        normalized = str(raw or "").strip().upper().strip("`\"' .,!?:;")
        if normalized in {"YES", "NO"}:
            return normalized
        match = re.search(r"\b(YES|NO)\b", normalized)
        return str(match.group(1) if match else "")

    def _run_execution_guard(
        self,
        *,
        session_id: str,
        response_signals: ResponseSignals | None = None,
    ) -> dict[str, Any] | None:
        ctx = self.memory.working.load(session_id)
        if ctx is None or ctx.state != TaskState.EXECUTION:
            return None

        total_steps = len(ctx.plan)
        done_steps = len(ctx.done)
        step_counter_done = bool(total_steps > 0 and done_steps >= total_steps)
        semantic_done = bool(
            response_signals
            and response_signals.has_code_artifact
            and response_signals.has_done_phrase
            and not response_signals.has_open_question
        )
        if not step_counter_done and not semantic_done:
            return None

        if step_counter_done:
            logger.info("[EXECUTION_GUARD] завершено по счётчику: %s/%s", done_steps, total_steps)
        if semantic_done:
            logger.info(
                "[EXECUTION_GUARD] завершено семантически: has_code=%s has_done=%s open_q=%s",
                response_signals.has_code_artifact if response_signals else False,
                response_signals.has_done_phrase if response_signals else False,
                response_signals.has_open_question if response_signals else False,
            )

        if semantic_done and total_steps > 0 and done_steps < total_steps:
            while True:
                fresh_ctx = self.memory.working.load(session_id)
                if fresh_ctx is None or fresh_ctx.state != TaskState.EXECUTION:
                    break
                if len(fresh_ctx.done) >= len(fresh_ctx.plan):
                    break
                try:
                    self.memory.working.complete_current_step(session_id=session_id)
                    updated_ctx = self.memory.working.load(session_id)
                    updated_done = len(updated_ctx.done) if updated_ctx else len(fresh_ctx.done) + 1
                    logger.info("[STEP_WRITTEN] current_step=%s сохранён в working layer", updated_done)
                except ValueError as exc:
                    logger.warning("[EXECUTION_GUARD] semantic completion sync failed: %s", exc)
                    break

        synced_ctx = self.memory.working.load(session_id)
        synced_done = len(synced_ctx.done) if synced_ctx else done_steps
        synced_total = len(synced_ctx.plan) if synced_ctx else total_steps
        if synced_total <= 0 or synced_done < synced_total:
            return None

        try:
            self.memory.working.request_validation(session_id=session_id)
            logger.info("[STATE_UPDATE] EXECUTION -> VALIDATION applied by execution_guard")
            return {
                "source": "execution_guard",
                "state_applied": True,
                "state_from": TaskState.EXECUTION.value,
                "state_to": TaskState.VALIDATION.value,
                "semantic_done": semantic_done,
            }
        except ValueError as exc:
            logger.warning("[EXECUTION_GUARD] transition blocked: %s", exc)
            return {
                "source": "execution_guard",
                "state_applied": False,
                "reason": str(exc),
                "semantic_done": semantic_done,
            }

    def _recover_invariant_fail_from_planning(
        self,
        *,
        session_id: str,
        user_id: str,
        user_message: str,
    ) -> dict[str, Any] | None:
        ctx = self.memory.working.load(session_id)
        if ctx is None or ctx.state != TaskState.PLANNING:
            return None
        if not ctx.plan:
            return None

        if ctx.current_step != ctx.plan[0]:
            ctx.current_step = ctx.plan[0]

        try:
            self.memory.working.transition_state(ctx, TaskState.EXECUTION)
            ctx.updated_at = datetime.utcnow().isoformat()
            self.memory.working.save(ctx)
            logger.info("[INVARIANT_RECOVERY] PLANNING -> EXECUTION after invariant_fail")
        except ValueError as exc:
            logger.warning("[INVARIANT_RECOVERY] transition blocked: %s", exc)
            return None

        messages, prompt_preview, read_meta = self.memory.build_messages(
            session_id=session_id,
            user_id=user_id,
            system_instructions=SYSTEM_PROMPT,
            data_context="",
            user_query=user_message,
        )
        self.last_prompt_preview = prompt_preview

        response = self._create_chat_completion(
            model=self.model,
            messages=messages,
            max_tokens=4096,
            temperature=0.7,
        )
        finish_reason = str(getattr(response.choices[0], "finish_reason", "") or "stop")
        raw_reply = response.choices[0].message.content or ""
        parsed = parse_response_markers(raw_reply)
        usage = getattr(response, "usage", None)
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)

        candidate_text = self._sanitize_reply_text(parsed.external)
        if not candidate_text:
            candidate_text = "Не удалось сформировать ответ. Уточните задачу или добавьте контекст."
        elif finish_reason == "length":
            candidate_text = candidate_text.rstrip() + "\n\n_Ответ обрезан по длине. Можно попросить продолжить._"

        assistant_message, protocol_meta = self._finalize_external_message(
            session_id=session_id,
            user_id=user_id,
            text=candidate_text,
            internal_trace=parsed.internal,
            raw_response=raw_reply,
            source="llm",
            last_user_message=user_message,
        )
        invariant_report = protocol_meta.get("invariant_report") or {}
        if invariant_report.get("overall_status") != "ok":
            finish_reason = "invariant_fail"
            assistant_message = self._build_invariant_failure_user_message(invariant_report)

        return {
            "assistant_message": assistant_message,
            "protocol_meta": protocol_meta,
            "finish_reason": finish_reason,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "read_meta": read_meta,
        }

    @staticmethod
    def _build_invariant_retry_prompt(report: dict[str, Any]) -> str:
        explanation = str(report.get("explanation") or "Нарушены инварианты.").strip()
        return (
            "Исправь предыдущий ответ и верни его в корректном формате:\n"
            "- обязательно <internal> и <external>\n"
            "- в <internal> включи маркеры [STEP_DONE: N], [NEXT_STATE: X], "
            "[VALIDATION_OK] или [VALIDATION_FAIL: причина]\n"
            "- не выводи маркеры в <external>\n"
            f"Причина исправления: {explanation}"
        )

    @staticmethod
    def _build_invariant_failure_user_message(report: dict[str, Any]) -> str:
        explanation = str(report.get("explanation") or "").strip()
        if not explanation:
            explanation = "Нарушены внутренние инварианты ответа."
        return (
            "Не удалось безопасно сформировать ответ в текущем виде.\n"
            f"Причина: {explanation}\n"
            "Сформулируйте запрос ещё раз, и я продолжу с корректным протоколом."
        )

    def _append_history(self, *, user_message: str, assistant_message: str) -> None:
        if self.ctx.active == "branching" and hasattr(self.ctx.strategy, "add_message"):
            self.ctx.strategy.add_message("user", user_message)
            self.ctx.strategy.add_message("assistant", assistant_message)
            self.conversation_history = list(self.ctx.strategy.build_context([]))
            return
        self.conversation_history.append({"role": "user", "content": user_message})
        self.conversation_history.append({"role": "assistant", "content": assistant_message})

    def _sanitize_reply_text(self, text: str) -> str:
        cleaned = str(text or "").replace("\r\n", "\n").strip()
        if not cleaned:
            return ""
        forbidden_markers = ["[PLAN]", "[ANALYTICS]", "[DIAGNOSIS]", "[ADVISORY]", "[CLARIFICATION]"]
        for marker in forbidden_markers:
            cleaned = cleaned.replace(marker, "")
        return "\n".join(line.rstrip() for line in cleaned.splitlines()).strip()

    def _enforce_hard_constraints_on_reply(self, *, user_id: str, reply: str) -> str:
        text = str(reply or "").strip()
        if not text:
            return ""
        constraints = self._get_active_hard_constraints(user_id=user_id)
        if not constraints:
            return text
        for constraint in constraints:
            reason = self._detect_constraint_violation(constraint=constraint, reply=text)
            if not reason:
                continue
            alternative = self._build_compliant_alternative(constraint)
            return (
                f"Не могу выполнить запрос в текущем виде: нарушается ограничение «{constraint}».\n"
                f"Почему: {reason}\n"
                f"Вместо этого предлагаю: {alternative}"
            )
        return text

    def _get_active_hard_constraints(self, *, user_id: str) -> list[str]:
        try:
            profile = self.memory.get_profile_snapshot(user_id=user_id) or {}
        except Exception as exc:
            logger.warning("[HARD_CONSTRAINTS] profile read skipped: %s", exc)
            return []
        payload = profile.get("hard_constraints") if isinstance(profile, dict) else {}
        if not isinstance(payload, dict) or not bool(payload.get("verified", False)):
            return []
        value = payload.get("value") if isinstance(payload, dict) else []
        if not isinstance(value, list):
            return []
        out: list[str] = []
        for item in value:
            text = str(item or "").strip()
            if text and text not in out:
                out.append(text)
        return out

    def _detect_constraint_violation(self, *, constraint: str, reply: str) -> str | None:
        rule = str(constraint or "").strip().lower()
        lower = str(reply or "").lower()
        if not rule:
            return None

        if "только swiftui" in rule:
            has_uikit = "import uikit" in lower or "uiviewcontroller" in lower or "uiview " in lower
            mentions_uikit = bool(re.search(r"\buikit\b", lower))
            negative_uikit = bool(re.search(r"(без|не использ|avoid)\s+uikit", lower))
            if has_uikit or (mentions_uikit and not negative_uikit):
                return "в ответе предлагается UIKit, хотя разрешён только SwiftUI."

        if "без сторонних зависим" in rule or "без внешних библиотек" in rule:
            dependency_markers = [
                "alamofire",
                "rxswift",
                "revenuecat",
                "firebase",
                "snapkit",
                "kingfisher",
                "sdwebimage",
                "lottie",
                "cocoapods",
                "carthage",
                "swift package manager",
                "spm",
            ]
            if any(marker in lower for marker in dependency_markers):
                return "в ответе предлагаются внешние библиотеки, что запрещено ограничением."

        if "без рекламы" in rule:
            ad_markers = ["admob", "ads", "баннер", "реклам"]
            has_ads = any(marker in lower for marker in ad_markers)
            says_no_ads = "без реклам" in lower
            if has_ads and not says_no_ads:
                return "в ответе присутствует рекламная модель, а она запрещена."

        if "mvvm" in rule:
            alt_arches = ["mvc", "viper", "clean architecture"]
            if any(token in lower for token in alt_arches) and "mvvm" not in lower:
                return "в ответе предлагается архитектура, отличная от MVVM."

        if "ios" in rule:
            max_version_match = re.search(r"ios\s*(\d+)", rule, flags=re.IGNORECASE)
            if max_version_match:
                max_version = int(max_version_match.group(1))
                versions = re.findall(r"ios\s*(\d+)", lower, flags=re.IGNORECASE)
                version_numbers = [int(v) for v in versions]
                if any(v > max_version for v in version_numbers):
                    return (
                        f"в ответе есть ориентация на iOS {max(version_numbers)}+, "
                        f"что конфликтует с ограничением iOS {max_version}."
                    )

        return None

    def _build_compliant_alternative(self, constraint: str) -> str:
        rule = str(constraint or "").strip().lower()
        if "только swiftui" in rule:
            return "вариант полностью на SwiftUI без UIKit."
        if "без сторонних зависим" in rule or "без внешних библиотек" in rule:
            return "реализацию только на стандартных Apple-фреймворках без сторонних пакетов."
        if "без рекламы" in rule:
            return "монетизацию через подписку или разовую покупку без рекламы."
        if "mvvm" in rule:
            return "структуру экранов и кода в архитектуре MVVM."
        if "ios" in rule:
            return "решение, совместимое с указанной версией iOS."
        return "совместимый вариант, который соблюдает заданные ограничения."

    @staticmethod
    def _is_memory_recall_request(message: str) -> bool:
        lower = str(message or "").strip().lower()
        if not lower:
            return False
        markers = [
            "что я у тебя спрашивал",
            "что я спрашивал",
            "о чем мы говорили",
            "о чём мы говорили",
            "напомни что я спрашивал",
            "какие вопросы я задавал",
            "что ты помнишь о нашем диалоге",
            "что ты помнишь из диалога",
            "перескажи диалог",
            "резюме диалога",
        ]
        return any(marker in lower for marker in markers)

    def _build_memory_recall_response(self) -> str:
        user_turns = [
            str(msg.get("content") or "").strip()
            for msg in self.conversation_history
            if str(msg.get("role") or "") == "user" and str(msg.get("content") or "").strip()
        ]
        if not user_turns:
            return "В этой сессии у меня пока нет сохранённых прошлых вопросов."
        recent = user_turns[-6:]
        lines = [f"{idx}. {text}" for idx, text in enumerate(recent, start=1)]
        prefix = "Вот что вы спрашивали в этой сессии:"
        if len(user_turns) > len(recent):
            prefix += " Показываю последние 6 вопросов."
        return prefix + "\n" + "\n".join(lines)

    def _estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        return self._estimate_cost_for_model(self.model, prompt_tokens, completion_tokens)

    def _estimate_cost_for_model(self, model_name: str, prompt_tokens: int, completion_tokens: int) -> float:
        pricing = self.COST_PER_1M.get(str(model_name or "").strip()) or self.COST_PER_1M.get(self.model)
        if not pricing:
            return 0.0
        in_cost = (float(prompt_tokens or 0) / 1_000_000.0) * float(pricing["input"])
        out_cost = (float(completion_tokens or 0) / 1_000_000.0) * float(pricing["output"])
        return in_cost + out_cost

    def _create_chat_completion(self, **kwargs: Any):
        request_kwargs = dict(kwargs)
        try:
            return self.llm_client.chat_completion(**request_kwargs)
        except Exception as exc:
            if not self._is_model_not_found_error(exc):
                raise

            requested_model = str(request_kwargs.get("model") or self.model).strip()
            fallback_model = self._pick_fallback_model(requested_model)
            if not fallback_model:
                raise

            logger.warning(
                "[MODEL] Requested model unavailable (%s), fallback to %s",
                requested_model,
                fallback_model,
            )
            self.set_model(fallback_model)
            request_kwargs["model"] = fallback_model
            return self.llm_client.chat_completion(**request_kwargs)

    @staticmethod
    def _is_model_not_found_error(exc: Exception) -> bool:
        code = str(getattr(exc, "code", "") or "").strip().lower()
        if code == "model_not_found":
            return True
        text = str(exc).lower()
        return "model_not_found" in text or "does not exist or you do not have access" in text

    def _pick_fallback_model(self, requested_model: str) -> str:
        current = str(requested_model or "").strip()
        for candidate in self.MODEL_FALLBACK_ORDER:
            if candidate == current:
                continue
            if candidate in self.COST_PER_1M:
                return candidate
        return ""

    @staticmethod
    def _load_env_if_needed() -> None:
        if os.getenv("OPENAI_API_KEY"):
            return
        env_path = ".env"
        if not os.path.exists(env_path):
            return
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for raw_line in f:
                    line = raw_line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, value = line.split("=", 1)
                    if key.strip() == "OPENAI_API_KEY":
                        os.environ["OPENAI_API_KEY"] = value.strip().strip('"').strip("'")
                        return
        except Exception:
            return

    @staticmethod
    def _pretty_json(payload: dict) -> str:
        try:
            return json.dumps(payload, ensure_ascii=False, indent=2, default=str)
        except Exception:
            return str(payload)
