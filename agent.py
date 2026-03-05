from __future__ import annotations

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
from storage import ensure_session

SYSTEM_PROMPT = """You are an iOS product engineering assistant.
Your mission is to help the user build and launch iOS apps from idea to monetization (subscriptions) and growth.
Primary stack: Swift + SwiftUI. Mention UIKit only when needed.
Final user-facing reply must be in Russian unless the user asks for another language.

Core capabilities:
- iOS architecture, Swift/SwiftUI implementation, testing, App Store release flow.
- Monetization: StoreKit 2, RevenueCat integration, paywall strategy, onboarding-to-conversion flow.
- Product iteration: analytics setup, conversion optimization, growth loops.

Response behavior:
- Think internally first, then provide user-facing answer.
- If clarification is needed, ask with numbered options (1, 2, 3) and ask user to reply with a number.
- Keep answers practical and code-oriented where relevant.
- End every user-facing answer with one clear next action sentence in natural language.

Output format (strict):
- Always return two sections:
  <internal>...hidden analysis, checks, state updates...</internal>
  <external>...human-friendly answer for the user...</external>
- Never include internal tags, metadata, state names, or invariant labels inside <external>.

Safety:
- Do not invent APIs that do not exist.
- Mention iOS/version constraints when relevant.
- Prefer robust maintainable solutions over quick hacks.
"""

logger = logging.getLogger("agent")


class IOSAgent:
    """iOS product assistant with memory layers and internal workflow orchestration."""

    DEFAULT_MODEL = "gpt-5.3-instant"
    DEFAULT_USER_ID = "default_local_user"
    MODEL_FALLBACK_ORDER = ("gpt-5-mini", "gpt-4o-mini")
    COST_PER_1M = {
        "gpt-5.3-instant": {"input": 1.75, "output": 14.00},
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

    def build_protocol_header(self, *, session_id: str, user_id: str, invariant_report: dict | None = None) -> dict:
        return self.memory.build_protocol_header(
            session_id=session_id,
            user_id=user_id,
            invariant_report=invariant_report,
        )

    def evaluate_invariants(self, *, run_external: bool = False) -> dict:
        return self.memory.evaluate_invariants(run_external=run_external)

    def prepare_protocol_turn(self, *, session_id: str, user_id: str, user_message: str) -> dict:
        return self.memory.prepare_protocol_turn(
            session_id=session_id,
            user_id=user_id,
            user_message=user_message,
        )

    def ensure_protocol_profile(self, *, user_id: str) -> dict:
        return self.memory.ensure_protocol_profile(user_id=user_id)

    def propose_profile_update(self, *, user_id: str, updates: dict, reason: str = "user_request") -> dict:
        return self.memory.propose_profile_update(user_id=user_id, updates=updates, reason=reason)

    def confirm_profile_update(self, *, user_id: str, accept: bool = True) -> dict:
        return self.memory.confirm_profile_update(user_id=user_id, accept=accept)

    def chat(self, user_message: str, session_id: str | None = None, user_id: str | None = None) -> str:
        """Handles one chat turn with memory-aware prompt building."""
        t_start = perf_counter()

        current_session_id = str(session_id or "default_session")
        current_user_id = str(user_id or self.DEFAULT_USER_ID)
        ensure_session(current_session_id)
        ctx_before_turn = self.memory.working.load(current_session_id)
        state_before_turn = ctx_before_turn.state if ctx_before_turn else None

        protocol_turn = self.prepare_protocol_turn(
            session_id=current_session_id,
            user_id=current_user_id,
            user_message=user_message,
        )

        gate_response = self._handle_planning_gate(
            session_id=current_session_id,
            user_id=current_user_id,
            user_message=user_message,
            started_at=t_start,
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

        state_before_route = state_after_gate
        self.memory.route_user_message(
            session_id=current_session_id,
            user_id=current_user_id,
            user_message=user_message,
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

        messages, prompt_preview, read_meta = self.memory.build_messages(
            session_id=current_session_id,
            user_id=current_user_id,
            system_instructions=SYSTEM_PROMPT,
            data_context=str(protocol_turn.get("internal_context") or ""),
            user_query=user_message,
        )
        self.last_prompt_preview = prompt_preview

        response = self._create_chat_completion(
            model=self.model,
            messages=messages,
            max_tokens=4096,
            temperature=0.7,
        )

        finish_reason = getattr(response.choices[0], "finish_reason", None)
        raw_reply = response.choices[0].message.content or ""
        internal_trace, external_text = self._split_internal_external(raw_reply)
        assistant_message = self._sanitize_reply_text(external_text)
        if not assistant_message:
            assistant_message = "Не удалось сформировать ответ. Уточните задачу или добавьте контекст."
        elif finish_reason == "length":
            assistant_message = assistant_message.rstrip() + "\n\n_Ответ обрезан по длине. Можно попросить продолжить._"
        assistant_message, protocol_meta = self._finalize_external_message(
            session_id=current_session_id,
            user_id=current_user_id,
            text=assistant_message,
            internal_trace=internal_trace,
            protocol_turn=protocol_turn,
        )

        self.memory.append_turn(
            session_id=current_session_id,
            user_message=user_message,
            assistant_message=assistant_message,
        )
        self._append_history(user_message=user_message, assistant_message=assistant_message)

        usage = getattr(response, "usage", None)
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        total_tokens = int(getattr(usage, "total_tokens", prompt_tokens + completion_tokens) or 0)
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
            "protocol_header": None,
            "protocol_state": protocol_meta.get("protocol_state"),
            "invariant_report": protocol_meta.get("invariant_report"),
            "next_step": protocol_meta.get("next_step"),
            "internal_trace": internal_trace,
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
        _ = messages
        self.memory.short_term.get_context(session_id)

    def _handle_planning_gate(
        self,
        *,
        session_id: str,
        user_id: str,
        user_message: str,
        started_at: float,
    ) -> str | None:
        guard_ctx_before = self.memory.working.load(session_id)
        gate_message = self.memory.enforce_planning_gate(
            session_id=session_id,
            user_message=user_message,
            user_id=user_id,
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
            goal = str(guard_ctx_before.task or "Текущая задача").strip() or "Текущая задача"
            if "утверж" in gate_message.lower():
                assistant_message = self._sanitize_reply_text(gate_message)
            else:
                assistant_message = self._sanitize_reply_text(
                    f"Задача создана: '{goal}'.\n"
                    "Чтобы перейти к выполнению, сначала сформируйте план.\n"
                    "Опишите шаги вручную или выберите действие автоматической генерации плана."
                )
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
            self.memory.set_protocol_state_meta(
                user_id=user_id,
                patch={
                    "awaiting_skip_confirmation": False,
                    "validation_skipped_by_user": False,
                    "validation_skip_note": "",
                },
            )
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
            "Если всё в порядке — нажмите «Подтвердить завершение».\n"
            "Если нужна доработка — нажмите «Вернуться к выполнению»."
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

        if ctx.state == TaskState.VALIDATION and self._is_checklist_request(msg):
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
    def _is_checklist_request(message: str) -> bool:
        lower = str(message or "").lower()
        return "чеклист" in lower or "checklist" in lower

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
    ) -> str:
        protocol_turn = self.prepare_protocol_turn(
            session_id=session_id,
            user_id=user_id,
            user_message=user_message,
        )
        formatted_message, protocol_meta = self._finalize_external_message(
            session_id=session_id,
            user_id=user_id,
            text=assistant_message,
            internal_trace="",
            protocol_turn=protocol_turn,
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
            "protocol_header": None,
            "protocol_state": protocol_meta.get("protocol_state"),
            "invariant_report": protocol_meta.get("invariant_report"),
            "next_step": protocol_meta.get("next_step"),
            "internal_trace": "",
        }
        return formatted_message

    def _finalize_external_message(
        self,
        *,
        session_id: str,
        user_id: str,
        text: str,
        internal_trace: str,
        protocol_turn: dict[str, Any] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        runtime = protocol_turn or self.prepare_protocol_turn(
            session_id=session_id,
            user_id=user_id,
            user_message="",
        )
        next_step = str(runtime.get("next_step") or "").strip()
        cleaned = self._strip_internal_artifacts(self._sanitize_reply_text(text))
        cleaned = self._enforce_hard_constraints_on_reply(user_id=user_id, reply=cleaned)
        if not cleaned:
            cleaned = "Готово, продолжаем."

        normalized = cleaned.rstrip()
        if next_step and next_step.lower() not in normalized.lower():
            if not re.search(r"[.!?]\s*$", normalized):
                normalized += "."
            normalized += f"\n\nЕсли ок, следующим шагом {next_step}."

        meta = {
            "protocol_state": runtime.get("protocol_state") or {},
            "invariant_report": runtime.get("invariant_report") or {},
            "next_step": next_step,
            "internal_trace": internal_trace,
        }
        return normalized.strip(), meta

    def _split_internal_external(self, text: str) -> tuple[str, str]:
        raw = str(text or "")
        if not raw.strip():
            return "", ""

        internal = ""
        external = raw

        internal_match = re.search(r"<internal>(.*?)</internal>", raw, flags=re.IGNORECASE | re.DOTALL)
        if internal_match:
            internal = str(internal_match.group(1) or "").strip()

        external_match = re.search(r"<external>(.*?)</external>", raw, flags=re.IGNORECASE | re.DOTALL)
        if external_match:
            external = str(external_match.group(1) or "").strip()
        else:
            external = re.sub(r"<internal>.*?</internal>", "", external, flags=re.IGNORECASE | re.DOTALL).strip()
            external = re.sub(r"</?external>", "", external, flags=re.IGNORECASE).strip()

        if not external:
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    external = str(parsed.get("external") or "").strip()
                    internal = str(parsed.get("internal") or internal).strip()
            except Exception:
                external = ""
        return internal, external or raw.strip()

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
            cleaned_lines.append(line)
        return "\n".join(cleaned_lines).strip()

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
            alt_arches = ["mvc", "viper", "mvp", "clean architecture"]
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

    def _estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        pricing = self.COST_PER_1M.get(self.model)
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
