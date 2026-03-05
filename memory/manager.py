from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime
import logging
import re
from typing import TYPE_CHECKING

from memory.long_term import LongTermMemory
from memory.models import MemoryWriteEvent, ProfileSource, TaskContext, TaskState
from memory.protocol import ProtocolCoordinator
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


class MemoryManager:
    VALIDATION_CONFIRMED_SIGNAL = "__VALIDATION_CONFIRMED__"

    def __init__(
        self,
        short_term_limit: int = 30,
        *,
        llm_client: "LLMClient | None" = None,
        step_parser_model: str = "gpt-5-nano",
    ):
        self.short_term = ShortTermMemory(limit_n=short_term_limit)
        self.working = WorkingMemory()
        self.long_term = LongTermMemory()
        self.router = MemoryRouter(llm_client=llm_client, step_parser_model=step_parser_model)
        self.prompt_builder = PromptBuilder()
        self.protocol = ProtocolCoordinator(long_term=self.long_term, working=self.working)
        self._write_events: dict[str, deque[dict]] = defaultdict(lambda: deque(maxlen=MAX_WRITE_EVENTS))

    def route_user_message(self, *, session_id: str, user_id: str, user_message: str):
        events = self.router.route_user_message(
            session_id=session_id,
            user_id=user_id,
            user_message=user_message,
            working=self.working,
            long_term=self.long_term,
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
        self.short_term.append(session_id=session_id, role="user", content=user_message)
        self.short_term.append(session_id=session_id, role="assistant", content=assistant_message)

    def clear_session(self, session_id: str) -> None:
        self.short_term.clear_session(session_id)
        self.working.clear_session(session_id)
        self._write_events.pop(session_id, None)

    def clear_working_layer(self, *, session_id: str) -> bool:
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

    def delete_long_term_entry(self, *, session_id: str, user_id: str, entry_type: str, entry_id: int) -> bool:
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
        self._sync_protocol_profile_to_canonical(
            user_id=user_id,
            reason="lazy_backfill_profile_read",
            session_id=session_id,
        )
        return self.long_term.get_profile(user_id=user_id)

    def ensure_protocol_profile(self, *, user_id: str) -> dict:
        return self.protocol.ensure_protocol_profile(user_id=user_id)

    def get_protocol_status(self, *, user_id: str) -> dict:
        return self.protocol.get_protocol_status(user_id=user_id)

    def propose_profile_update(self, *, user_id: str, updates: dict, reason: str = "user_request") -> dict:
        return self.protocol.propose_profile_update(user_id=user_id, updates=updates, reason=reason)

    def confirm_profile_update(self, *, user_id: str, accept: bool = True) -> dict:
        return self.protocol.confirm_profile_update(user_id=user_id, accept=accept)

    def set_protocol_state_meta(self, *, user_id: str, patch: dict) -> dict:
        return self.protocol.set_protocol_state_meta(user_id=user_id, patch=patch)

    def build_protocol_header(
        self,
        *,
        session_id: str,
        user_id: str,
        invariant_report: dict | None = None,
    ) -> dict:
        return self.protocol.build_protocol_header(
            session_id=session_id,
            user_id=user_id,
            invariant_report=invariant_report,
        )

    def evaluate_invariants(self, *, run_external: bool = False) -> dict:
        return self.protocol.evaluate_invariants(run_external=run_external)

    def prepare_protocol_turn(self, *, session_id: str, user_id: str, user_message: str) -> dict:
        runtime = self.protocol.prepare_turn(
            session_id=session_id,
            user_id=user_id,
            user_message=user_message,
        )
        runtime["canonical_sync"] = self._sync_protocol_profile_to_canonical(
            user_id=user_id,
            reason="prepare_protocol_turn",
            session_id=session_id,
        )
        return runtime

    def _sync_protocol_profile_to_canonical(
        self,
        *,
        user_id: str,
        reason: str,
        session_id: str | None,
    ) -> dict:
        snapshot = self.protocol.ensure_protocol_profile(user_id=user_id)
        protocol_profile = dict(snapshot.get("protocol_profile") or {})
        current_profile = self.long_term.get_profile(user_id=user_id) or {}
        patch = self.protocol.build_canonical_patch(
            protocol_profile=protocol_profile,
            current_profile=current_profile,
        )

        updated_fields: list[str] = []
        conflict_fields: list[str] = []
        skipped_fields: list[str] = []
        errors: dict[str, str] = {}

        for field, payload in patch.items():
            value = payload.get("value") if isinstance(payload, dict) else None
            confidence_raw = payload.get("confidence") if isinstance(payload, dict) else None
            try:
                confidence = float(confidence_raw or 0.0)
            except Exception:
                confidence = 0.0
            if confidence <= 0.0:
                skipped_fields.append(field)
                continue

            field_payload = current_profile.get(field) if isinstance(current_profile, dict) else {}
            current_value = field_payload.get("value") if isinstance(field_payload, dict) else None
            if self._profile_values_equal(field=field, left=current_value, right=value):
                skipped_fields.append(field)
                continue

            if self._has_open_conflict(
                profile=current_profile,
                field=field,
                existing_value=current_value,
                inferred_value=value,
            ):
                skipped_fields.append(field)
                continue

            try:
                status = self.long_term.update_profile_field(
                    user_id=user_id,
                    field=field,
                    value=value,
                    source=ProfileSource.AGENT_INFERRED,
                    confidence=confidence,
                    verified=False,
                )
            except Exception as exc:
                errors[field] = str(exc)
                logger.warning("[PROFILE_SYNC_SKIP] field=%s reason=%s", field, exc)
                continue

            if status == "updated":
                updated_fields.append(field)
                current_profile = self.long_term.get_profile(user_id=user_id) or current_profile
            elif status == "conflict_recorded":
                conflict_fields.append(field)
                current_profile = self.long_term.get_profile(user_id=user_id) or current_profile
            else:
                skipped_fields.append(field)

        keys_to_record = list(dict.fromkeys(updated_fields + conflict_fields + (["conflicts"] if conflict_fields else [])))
        if session_id and keys_to_record:
            self._record_write_event(
                session_id=session_id,
                layer="long_term.profile",
                keys=keys_to_record,
                operation="save",
                source="protocol_sync",
            )

        if updated_fields:
            status = "updated"
        elif conflict_fields:
            status = "conflict_recorded"
        elif errors:
            status = "failed"
        else:
            status = "no_changes"

        return {
            "status": status,
            "reason": str(reason or ""),
            "updated_fields": updated_fields,
            "conflict_fields": conflict_fields,
            "skipped_fields": skipped_fields,
            "errors": errors,
        }

    def debug_update_profile_field(
        self,
        *,
        session_id: str,
        user_id: str,
        field: str,
        value: object,
    ) -> dict:
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
        self.short_term.hydrate(session_id=session_id, messages=messages)

    def enforce_planning_gate(self, *, session_id: str, user_message: str, user_id: str | None = None) -> str | None:
        ctx: TaskContext | None = self.working.load(session_id)
        if not ctx:
            return None

        msg = (user_message or "").strip().lower()
        status = self.working.get_step_status(ctx)
        step_index = status.get("step_index")
        total_steps = status.get("total_steps")

        if ctx.state == TaskState.PLANNING:
            plan_empty = self._plan_is_empty(ctx)
            if self._is_skip_request(msg) and plan_empty:
                return "Сначала сформируйте план задачи, затем можно перейти к выполнению."
            if bool((ctx.vars or {}).get("plan_guidance_required")):
                return "Опишите шаги плана или нажмите 'Сформировать план автоматически'."
            if plan_empty and self._is_start_execution_request(msg):
                return "Сначала сформируйте план задачи, затем можно перейти к выполнению."
            if plan_empty or self._is_plan_formation_message(msg) or self._is_goal_clarification_message(msg):
                return None

            if self._is_plan_approved_message(msg):
                if not ctx.plan:
                    return "Сначала сформируйте план задачи, затем подтвердите его."
                if ctx.current_step != ctx.plan[0]:
                    return "Перед стартом работы текущий шаг должен совпадать с первым шагом плана."
                try:
                    self.working.transition_state(ctx, TaskState.EXECUTION)
                    ctx.updated_at = datetime.utcnow().isoformat()
                    self.working.save(ctx)
                except ValueError as exc:
                    return str(exc)
                return None

            wants_start = self._is_start_execution_request(msg)
            if wants_start and ctx.plan:
                return "План уже готов. Напишите «план утверждён», и начнём реализацию."
            return "Ожидается план и явное подтверждение: «план утверждён»."

        if ctx.state == TaskState.EXECUTION:
            all_steps_done = bool(ctx.current_step is None and ctx.done == ctx.plan and ctx.plan)
            vars_patch = dict(ctx.vars or {})
            awaiting_skip = bool(vars_patch.get("awaiting_validation_skip_confirmation"))

            if awaiting_skip:
                if self._is_yes_confirmation(msg):
                    vars_patch["awaiting_validation_skip_confirmation"] = False
                    vars_patch["allow_validation_skip"] = True
                    vars_patch["validation_skipped_by_user"] = True
                    vars_patch["validation_skip_note"] = "⚠️ VALIDATION SKIPPED BY USER (риск на пользователе)"
                    self.working.update(session_id, vars=vars_patch)
                    updated_ctx = self.working.load(session_id)
                    if not updated_ctx:
                        return "working task not found"
                    try:
                        self.working.transition_state(updated_ctx, TaskState.DONE)
                        updated_ctx.updated_at = datetime.utcnow().isoformat()
                        self.working.save(updated_ctx)
                    except ValueError as exc:
                        return str(exc)
                    if user_id:
                        self.set_protocol_state_meta(
                            user_id=user_id,
                            patch={
                                "awaiting_skip_confirmation": False,
                                "validation_skipped_by_user": True,
                                "validation_skip_note": "⚠️ VALIDATION SKIPPED BY USER (риск на пользователе)",
                            },
                        )
                    return "Понял, пропускаем отдельный этап проверки и завершаем задачу."

                if self._is_no_confirmation(msg):
                    vars_patch["awaiting_validation_skip_confirmation"] = False
                    vars_patch.pop("allow_validation_skip", None)
                    self.working.update(session_id, vars=vars_patch)
                    if user_id:
                        self.set_protocol_state_meta(
                            user_id=user_id,
                            patch={"awaiting_skip_confirmation": False},
                        )
                    return "Пропуск проверки отменён. Тогда переходим к обычной финальной проверке."

                return "Проверка ещё не завершена. Пропустить этот этап и перейти дальше? Ответьте yes или no."

            if all_steps_done:
                if self._is_validation_skip_request(msg):
                    vars_patch["awaiting_validation_skip_confirmation"] = True
                    self.working.update(session_id, vars=vars_patch)
                    if user_id:
                        self.set_protocol_state_meta(
                            user_id=user_id,
                            patch={"awaiting_skip_confirmation": True},
                        )
                    return "Все шаги готовы. Нужна финальная проверка. Пропустить её и завершить? Ответьте yes или no."
                try:
                    self.working.request_validation(session_id)
                    return None
                except ValueError as exc:
                    return str(exc)

            if self._is_validation_skip_request(msg):
                return "Пропуск финальной проверки возможен только после завершения всех шагов плана."

            if self._is_validation_request(msg):
                try:
                    self.working.request_validation(session_id)
                    return "Отлично, переходим к финальной проверке результата."
                except ValueError as exc:
                    return str(exc)

            if self._is_execution_allowed_message(msg, ctx.current_step or ""):
                return None
            return (
                f"Сейчас выполняется шаг {step_index}/{total_steps}: '{ctx.current_step}'. "
                "Завершите его перед сменой контекста."
            )

        if ctx.state == TaskState.VALIDATION:
            if "чеклист" in msg or "checklist" in msg:
                return None
            if self._is_validation_confirm_message(msg):
                return self.VALIDATION_CONFIRMED_SIGNAL
            if self._is_validation_reject_message(msg):
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
        ctx = self.working.load(session_id)
        if not ctx:
            return {
                "state": None,
                "current_step": None,
                "step_index": None,
                "total_steps": 0,
                "done": [],
                "plan": [],
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
        }

    def get_working_actions(self, *, session_id: str) -> list[dict]:
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
        working = self.working.load(session_id)
        longterm = self.long_term.retrieve(user_id=user_id, query="", top_k=3)
        read_meta = dict(longterm.get("read_meta") or {})
        profile = longterm.get("profile") or {}
        has_profile = bool(
            (profile.get("stack_tools") or {}).get("value")
            or (profile.get("response_style") or {}).get("value")
            or (profile.get("hard_constraints") or {}).get("value")
            or (profile.get("user_role_level") or {}).get("value")
            or (profile.get("project_context") or {}).get("value")
            or (profile.get("extra_fields") or {})
            or (profile.get("conflicts") or [])
        )
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
        return self._normalize_profile_value(field=field, value=left) == self._normalize_profile_value(field=field, value=right)

    def _normalize_profile_value(self, *, field: str, value: object) -> object:
        if field in {"stack_tools", "hard_constraints"}:
            return self._normalize_text_list(value)
        if field in {"response_style", "user_role_level"}:
            return str(value or "").strip()
        if field == "project_context":
            return self._normalize_project_context_value(value)
        return value

    @staticmethod
    def _normalize_text_list(value: object) -> list[str]:
        if not isinstance(value, list):
            return []
        out: list[str] = []
        for item in value:
            text = str(item or "").strip()
            if text and text not in out:
                out.append(text)
        return out

    def _normalize_project_context_value(self, value: object) -> dict:
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
        for event in events:
            self._record_write_event(
                session_id=session_id,
                layer=event.layer,
                keys=event.keys,
                operation="save",
                source=source,
            )

    def get_recent_write_events(self, *, session_id: str, limit: int = 10) -> list[dict]:
        events = list(self._write_events.get(session_id) or [])
        lim = max(1, int(limit))
        return events[-lim:]

    def _format_entry_for_debug(self, entry: dict) -> dict:
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
        """Снимок трёх слоёв памяти для Debug UI. top_k в диапазоне 1..10."""
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
                "decisions_top_k": decisions_top_k,
                "notes_top_k": notes_top_k,
                "read_meta": read_meta,
            },
            "memory_writes": self.get_recent_write_events(session_id=session_id, limit=10),
        }

    def _is_start_execution_request(self, msg: str) -> bool:
        triggers = [
            "начинаем выполнение",
            "перейди к выполнению",
            "start execution",
            "run plan",
            "go execution",
        ]
        return any(trigger in msg for trigger in triggers)

    def _is_plan_approved_message(self, msg: str) -> bool:
        triggers = ["план утверждён", "план утвержден", "plan approved"]
        return any(trigger in msg for trigger in triggers)

    def _is_plan_formation_message(self, msg: str) -> bool:
        if not msg:
            return False
        patterns = [
            r"\bсформируй план\b",
            r"\bсостав(?:ь|ьте)? план\b",
            r"\bразбей на шаги\b",
            r"\bкакие шаги\b",
            r"\bплан задачи\b",
            r"\bс чего начать\b",
            r"\bавтоматически\b",
            r"\bform plan\b",
            r"\bcreate plan\b",
        ]
        return any(re.search(pattern, msg, flags=re.IGNORECASE) for pattern in patterns)

    def _is_skip_request(self, msg: str) -> bool:
        if not msg:
            return False
        patterns = [
            r"\bдай финальн",
            r"\bсразу результат\b",
            r"\bпропусти планирован",
            r"\bskip\b",
            r"\bфинальный ответ сейчас\b",
            r"\bсразу к делу\b",
        ]
        return any(re.search(pattern, msg, flags=re.IGNORECASE) for pattern in patterns)

    def _is_goal_clarification_message(self, msg: str) -> bool:
        if not msg:
            return False
        patterns = [
            r"\bуточним цель\b",
            r"\bуточнить цель\b",
            r"\bцель задачи\b",
            r"\bуточни\b.*\bцель\b",
            r"\bclarify goal\b",
        ]
        return any(re.search(pattern, msg, flags=re.IGNORECASE) for pattern in patterns)

    @staticmethod
    def _plan_is_empty(ctx: TaskContext) -> bool:
        return len(list(ctx.plan or [])) == 0

    def _is_validation_request(self, msg: str) -> bool:
        triggers = [
            "валидац",
            "провер",
            "подтверди шаги",
            "request_validation",
            "перейди в validation",
        ]
        return any(trigger in msg for trigger in triggers)

    def _is_validation_confirm_message(self, msg: str) -> bool:
        return any(trigger in msg for trigger in self.router.VALIDATION_CONFIRM_PATTERNS)

    def _is_validation_reject_message(self, msg: str) -> bool:
        return any(trigger in msg for trigger in self.router.VALIDATION_REJECT_PATTERNS)

    def _is_validation_skip_request(self, msg: str) -> bool:
        triggers = ["пропусти validation", "skip validation", "без validation", "пропустить validation"]
        return any(trigger in msg for trigger in triggers)

    def _is_yes_confirmation(self, msg: str) -> bool:
        normalized = msg.strip().lower()
        return normalized in {"yes", "y", "да", "ок", "подтверждаю"}

    def _is_no_confirmation(self, msg: str) -> bool:
        normalized = msg.strip().lower()
        return normalized in {"no", "n", "нет", "отмена", "не подтверждаю"}

    def _is_execution_allowed_message(self, msg: str, current_step: str) -> bool:
        return self.router.is_execution_allowed_message(text=msg, current_step=current_step)
