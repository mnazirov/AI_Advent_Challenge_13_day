from __future__ import annotations

import ast
import json
import logging
import re
from typing import TYPE_CHECKING, Any, Callable

from memory.intents import IntentDecision, IntentName, parse_client_intent
from memory.long_term import LongTermMemory
from memory.models import ArtifactType, MemoryWriteEvent, TaskState
from memory.working import WorkingMemory

if TYPE_CHECKING:
    from llm.client import LLMClient

logger = logging.getLogger("memory")


class MemoryRouter:
    WORKING_CONFIDENCE_THRESHOLD = 0.65
    PROFILE_FORBIDDEN_SOURCES = ["working_memory", "task_artifact", "step_result"]

    def __init__(self, *, llm_client: "LLMClient | None" = None, step_parser_model: str = "gpt-5-nano"):
        self.llm_client = llm_client
        self.step_parser_model = str(step_parser_model or "gpt-5-nano")
        self._intent_cache: dict[tuple[str, str, str], IntentDecision] = {}
        self._aux_llm_budget_reserver: Callable[[str], bool] | None = None
        self._last_working_extract_meta: dict[str, dict[str, Any]] = {}

    def set_aux_llm_budget_reserver(self, fn: Callable[[str], bool] | None) -> None:
        """Устанавливает callback-резервер слота для служебных LLM-вызовов."""
        self._aux_llm_budget_reserver = fn

    def get_last_working_extract_meta(self, session_id: str) -> dict[str, Any]:
        """Возвращает метаданные последнего working_extract для сессии."""
        sid = str(session_id or "")
        meta = self._last_working_extract_meta.get(sid) or {}
        return dict(meta)

    def route_user_message(
        self,
        *,
        session_id: str,
        user_id: str,
        user_message: str,
        working: WorkingMemory,
        long_term: LongTermMemory,
        client_intent: dict[str, Any] | None = None,
    ) -> list[MemoryWriteEvent]:
        text = (user_message or "").strip()
        lower = text.lower()
        events: list[MemoryWriteEvent] = []
        structured_intent, structured_payload = parse_client_intent(client_intent)

        pending_id = self._extract_pending_confirmation_id(
            structured_intent=structured_intent,
            payload=structured_payload,
        )
        if pending_id is not None:
            self._guard_profile_source("explicit_confirmation")
            approved = long_term.approve_pending_entry(user_id=user_id, pending_id=pending_id)
            if approved:
                events.append(MemoryWriteEvent(layer="long_term.approval", keys=["pending_id", "type"]))
            else:
                logger.info("[MEMORY_WRITE] layer=long_term.approval status=not_found pending_id=%s", pending_id)
            return events

        decision_intent = self._decision_intent(text=text, structured_intent=structured_intent)
        if decision_intent.is_unknown:
            logger.info(
                "[INTENT_ROUTE] intent=%s status=%s reason_code=%s",
                decision_intent.intent.value,
                decision_intent.status,
                decision_intent.reason_code,
            )
        if decision_intent.is_match:
            tags = ["decision"]
            if "стандарт" in lower:
                tags.append("standard")
            result = long_term.add_decision(user_id=user_id, text=text, tags=tags, source="user")
            if result.get("status") == "pending":
                events.append(MemoryWriteEvent(layer="long_term.pending", keys=["pending_id"]))
            else:
                events.append(MemoryWriteEvent(layer="long_term.decision", keys=["text", "tags"]))

        note_intent = self._note_intent(text=text, structured_intent=structured_intent)
        if note_intent.is_unknown:
            logger.info(
                "[INTENT_ROUTE] intent=%s status=%s reason_code=%s",
                note_intent.intent.value,
                note_intent.status,
                note_intent.reason_code,
            )
        if note_intent.is_match:
            tags = ["note", "stability"]
            result = long_term.add_note(
                user_id=user_id,
                text=text,
                tags=tags,
                source="user",
                ttl_days=90,
            )
            if result.get("status") == "pending":
                events.append(MemoryWriteEvent(layer="long_term.pending", keys=["pending_id"]))
            else:
                events.append(MemoryWriteEvent(layer="long_term.note", keys=["text", "tags", "ttl_days"]))

        existing_ctx = working.load(session_id)
        plan_formation_intent = self._plan_formation_intent(text=text, structured_intent=structured_intent)
        task_intent = self._task_intent(text=text, structured_intent=structured_intent)
        if plan_formation_intent.is_unknown:
            logger.info(
                "[INTENT_ROUTE] intent=%s status=%s reason_code=%s",
                plan_formation_intent.intent.value,
                plan_formation_intent.status,
                plan_formation_intent.reason_code,
            )
        if task_intent.is_unknown:
            logger.info(
                "[INTENT_ROUTE] intent=%s status=%s reason_code=%s",
                task_intent.intent.value,
                task_intent.status,
                task_intent.reason_code,
            )

        if existing_ctx is None and self._should_auto_start_task_context(
            text=text,
            structured_intent=structured_intent,
        ):
            auto_goal = self._extract_goal(text)
            existing_ctx = working.start_task(session_id=session_id, goal=auto_goal)
            events.append(MemoryWriteEvent(layer="working", keys=["task", "state"]))
            logger.info('[TASK_AUTO_START] goal="%s" session=%s', auto_goal, session_id)
        existing_state = existing_ctx.state if existing_ctx else TaskState.PLANNING

        working_patch = self._extract_working_patch_from_client_intent(
            structured_intent=structured_intent,
            payload=structured_payload,
            existing_current_step=(existing_ctx.current_step if existing_ctx else None),
        )
        llm_applied = False
        llm_confidence = 0.0
        llm_keys: list[str] = []
        working_extract_reason_code = "working_extract_structured_noop"

        if not self._working_patch_has_changes(working_patch):
            llm_patch, working_extract_reason_code = self._extract_working_patch_via_llm(
                text=text,
                current_plan=list(existing_ctx.plan) if existing_ctx else [],
                current_step=existing_ctx.current_step if existing_ctx else None,
                done_steps=list(existing_ctx.done) if existing_ctx else [],
                working_state=(existing_ctx.state.value if existing_ctx else "NONE"),
                goal=(existing_ctx.task if existing_ctx else self._extract_goal(text)),
            )
            llm_confidence = float(llm_patch.get("confidence", 0.0)) if llm_patch else 0.0
            llm_keys = self._working_patch_keys(llm_patch) if llm_patch else []
            if (
                llm_patch
                and bool(llm_patch.get("is_working_update"))
                and llm_confidence >= self.WORKING_CONFIDENCE_THRESHOLD
                and self._working_patch_has_changes(llm_patch)
            ):
                working_patch = self._normalize_working_patch_payload(llm_patch)
                llm_applied = True
        if (
            not self._working_patch_has_changes(working_patch)
            and existing_ctx is not None
            and existing_ctx.state == TaskState.PLANNING
            and not list(existing_ctx.plan or [])
        ):
            fallback_plan = self._build_fallback_plan(goal=existing_ctx.task or text)
            working_patch = self._empty_working_patch()
            working_patch["is_working_update"] = True
            working_patch["plan"] = list(fallback_plan)
            working_patch["current_step"] = fallback_plan[0] if fallback_plan else ""
            llm_applied = False
            llm_confidence = max(llm_confidence, 0.61)
            llm_keys = ["plan", "current_step"]
            working_extract_reason_code = "working_extract_fallback_plan"
        logger.info(
            "[MEMORY_WORKING_EXTRACT] source=%s applied=%s confidence=%.2f keys=%s reason_code=%s",
            "llm" if llm_applied else "none",
            llm_applied,
            llm_confidence,
            ",".join(llm_keys) if llm_keys else "-",
            working_extract_reason_code,
        )
        self._last_working_extract_meta[str(session_id)] = {
            "applied": bool(llm_applied),
            "confidence": float(llm_confidence),
            "reason_code": str(working_extract_reason_code or ""),
        }

        if self._working_patch_has_changes(working_patch):
            ctx = existing_ctx or working.ensure_task(session_id=session_id, goal="Текущая задача")
            state = ctx.state
            try:
                if state == TaskState.PLANNING:
                    changed_keys = self._apply_planning_patch(
                        working=working,
                        session_id=session_id,
                        ctx=ctx,
                        patch=working_patch,
                    )
                    if changed_keys:
                        events.append(MemoryWriteEvent(layer="working", keys=changed_keys))
                elif state == TaskState.EXECUTION:
                    changed_keys = self._apply_execution_patch(
                        working=working,
                        session_id=session_id,
                        ctx=ctx,
                        patch=working_patch,
                    )
                    if changed_keys:
                        events.append(MemoryWriteEvent(layer="working", keys=changed_keys))
                else:
                    logger.info(
                        "[MEMORY_WRITE] layer=working blocked=true state=%s reason=writes_forbidden_in_state",
                        state.value,
                    )
            except ValueError as exc:
                logger.info(
                    "[MEMORY_WRITE] layer=working blocked=true state=%s reason=%s",
                    existing_state.value,
                    str(exc),
                )

        if not events:
            logger.info("[MEMORY_WRITE] layer=none reason=no_policy_match")
        else:
            for event in events:
                logger.info("[MEMORY_WRITE] layer=%s keys=%s", event.layer, ",".join(event.keys))

        return events

    def is_execution_allowed_message(self, *, text: str, current_step: str) -> bool:
        normalized = str(text or "").strip()
        step = str(current_step or "").strip()
        if not normalized:
            return False
        if self.llm_client is None:
            return True
        if not self._reserve_aux_llm_slot("execution_message_policy"):
            return True

        prompt = f"""You classify whether a user message is allowed in EXECUTION state of a task-state machine.
Return ONLY valid JSON.

Schema:
{{
  "allow": true/false,
  "confidence": 0.0,
  "reason": "short"
}}

Rules:
- allow=false only when the message explicitly asks to switch to another unrelated task/topic, or to bypass mandatory process control.
- allow=true for all implementation-related requests, including broader plan/architecture clarifications, current-step help, code requests, or neutral continuation messages.
- Keep confidence in [0,1].

Current step:
{step}

User message:
{normalized}
"""
        try:
            response = self.llm_client.chat_completion(
                model=self.step_parser_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=120,
                response_format={"type": "json_object"},
            )
            raw = str(response.choices[0].message.content or "").strip()
            payload = self._extract_first_json_object(raw)
            if not isinstance(payload, dict):
                return True
            allow = bool(payload.get("allow"))
            confidence = self._clamp_confidence(payload.get("confidence"))
            if confidence < 0.6:
                return True
            return allow
        except Exception as exc:
            logger.warning("[EXECUTION_POLICY] llm classify failed: %s", exc)
            return True

    def _extract_working_patch_from_client_intent(
        self,
        *,
        structured_intent: IntentName | None,
        payload: dict[str, Any],
        existing_current_step: str | None,
    ) -> dict[str, Any]:
        patch = self._empty_working_patch()
        if structured_intent is None:
            return patch
        if structured_intent == IntentName.STEP_COMPLETED and existing_current_step:
            patch["done_steps_to_add"] = [str(existing_current_step)]
        elif structured_intent == IntentName.WORKING_UPDATE and isinstance(payload, dict):
            patch = self._normalize_working_patch_payload(payload)

        patch["is_working_update"] = self._working_patch_has_changes(patch)
        return patch

    def _extract_working_patch_via_llm(
        self,
        *,
        text: str,
        current_plan: list[str],
        current_step: str | None,
        done_steps: list[str],
        working_state: str,
        goal: str,
    ) -> tuple[dict[str, Any], str]:
        if not text:
            return {}, "working_extract_empty_message"
        if self.llm_client is None:
            return {}, "intent_unknown_unavailable"
        if not self._reserve_aux_llm_slot("working_extract"):
            return {}, "intent_unknown_budget"

        prompt = f"""You extract working-memory task updates from a single user message.
Return ONLY valid JSON with no markdown and no explanations.

Expected JSON schema:
{{
  "is_working_update": true/false,
  "task": "...",
  "plan": ["..."],
  "plan_steps_to_add": ["..."],
  "current_step": "...",
  "done_steps_to_add": ["..."],
  "requirements_to_add": ["..."],
  "artifacts_to_add": ["..."],
  "confidence": 0.0
}}

Rules:
- If message is not about active task progress/planning, return is_working_update=false and empty fields.
- Keep arrays short and deduplicated.
- plan should contain ordered steps (min 2, max 10) when user asks to form a plan.
- confidence is mandatory in every response JSON.
- Keep confidence in [0,1].
- If user asks to make a plan or analysis, set is_working_update=true, confidence=0.85, and fill task with a short goal.
- If working_state=PLANNING and user asks to form a plan for task '{goal}', infer ordered plan steps and set current_step to step 1.
- If user message contains plan-formation intent, always set confidence >= 0.85.

Example for task intent:
{{
  "is_working_update": true,
  "task": "Составить детальный план реализации iOS-фичи",
  "plan": [
    "Уточнить функциональные и UX-требования",
    "Разбить реализацию на шаги по SwiftUI/архитектуре",
    "Добавить тест-кейсы и критерии приемки",
    "Проверить совместимость с целевой версией iOS"
  ],
  "plan_steps_to_add": [],
  "current_step": "Уточнить функциональные и UX-требования",
  "done_steps_to_add": [],
  "requirements_to_add": [],
  "artifacts_to_add": [],
  "confidence": 0.85
}}

Current context:
- working_state={working_state}
- goal={goal}
- plan={current_plan}
- current_step={current_step}
- done_steps={done_steps}

User message:
{text}
"""

        try:
            response = self.llm_client.chat_completion(
                model=self.step_parser_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=350,
                response_format={"type": "json_object"},
            )
            raw, payload = self._extract_response_payload(
                response=response,
                required_keys={"is_working_update", "confidence"},
            )
            logger.info("[MEMORY_WORKING_EXTRACT_RAW] response=%s", raw)
            if not isinstance(payload, dict):
                return {}, "working_extract_invalid_payload"
            return self._normalize_working_patch_payload(payload), "working_extract_ok"
        except Exception as exc:
            logger.warning("[MEMORY_WORKING_EXTRACT] source=llm parse_error=%s", exc)
            return {}, "working_extract_error"

    def _normalize_working_patch_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "is_working_update": bool(payload.get("is_working_update")),
            "task": str(payload.get("task") or "").strip(),
            "plan": self._normalize_str_list(payload.get("plan")),
            "plan_steps_to_add": self._normalize_str_list(payload.get("plan_steps_to_add")),
            "current_step": str(payload.get("current_step") or "").strip(),
            "done_steps_to_add": self._normalize_str_list(payload.get("done_steps_to_add")),
            "requirements_to_add": self._normalize_str_list(payload.get("requirements_to_add")),
            "artifacts_to_add": self._normalize_str_list(payload.get("artifacts_to_add")),
            "confidence": self._clamp_confidence(payload.get("confidence")),
        }

    def _working_patch_has_changes(self, patch: dict[str, Any]) -> bool:
        if not patch:
            return False
        return bool(
            patch.get("plan")
            or
            patch.get("current_step")
            or patch.get("plan_steps_to_add")
            or patch.get("done_steps_to_add")
            or patch.get("requirements_to_add")
            or patch.get("artifacts_to_add")
        )

    def _working_patch_keys(self, patch: dict[str, Any]) -> list[str]:
        if not patch:
            return []
        keys: list[str] = []
        for key in [
            "task",
            "plan",
            "plan_steps_to_add",
            "current_step",
            "done_steps_to_add",
            "requirements_to_add",
            "artifacts_to_add",
        ]:
            val = patch.get(key)
            if (isinstance(val, list) and val) or (isinstance(val, str) and val.strip()):
                keys.append(key)
        return keys

    def _empty_working_patch(self) -> dict[str, Any]:
        return {
            "is_working_update": False,
            "task": "",
            "plan": [],
            "plan_steps_to_add": [],
            "current_step": "",
            "done_steps_to_add": [],
            "requirements_to_add": [],
            "artifacts_to_add": [],
            "confidence": 0.0,
        }

    def _pick_first_pending_step(self, *, plan: list[str], done_steps: list[str]) -> str:
        done = {str(x) for x in (done_steps or [])}
        for step in plan:
            s = str(step)
            if s not in done:
                return s
        return str(plan[0]) if plan else ""

    def _normalize_str_list(self, value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        out: list[str] = []
        for item in value:
            s = str(item or "").strip()
            if s and s not in out:
                out.append(s)
        return out

    def _clamp_confidence(self, value: Any) -> float:
        try:
            num = float(value)
        except Exception:
            num = 0.0
        return max(0.0, min(1.0, num))

    def _append_unique(self, target: list[str], values: list[str]) -> None:
        for val in values:
            s = str(val or "").strip()
            if s and s not in target:
                target.append(s)

    def _task_intent(self, *, text: str, structured_intent: IntentName | None = None) -> IntentDecision:
        decision = self._classify_intent_llm(
            intent=IntentName.TASK_INTENT,
            text=text,
            guideline="User asks to create/start a concrete task or project goal.",
            structured_intent=structured_intent,
        )
        if decision.is_match or structured_intent is not None:
            return decision
        if decision.is_unknown:
            recovered = self._recover_intent_with_separate_analysis(
                intent=IntentName.TASK_INTENT,
                text=text,
                guideline="User asks to create/start a concrete task or project goal.",
            )
            if recovered is not None:
                return recovered
        return decision

    def _is_task_intent(self, text: str) -> bool:
        return self._task_intent(text=text).is_match

    def _plan_formation_intent(self, *, text: str, structured_intent: IntentName | None = None) -> IntentDecision:
        decision = self._classify_intent_llm(
            intent=IntentName.PLAN_FORMATION_INTENT,
            text=text,
            guideline="User asks to generate/build/refine an implementation plan.",
            structured_intent=structured_intent,
        )
        if decision.is_match or structured_intent is not None:
            return decision
        if decision.is_unknown:
            recovered = self._recover_intent_with_separate_analysis(
                intent=IntentName.PLAN_FORMATION_INTENT,
                text=text,
                guideline="User asks to generate/build/refine an implementation plan.",
            )
            if recovered is not None:
                return recovered
        return decision

    def _is_plan_formation_intent(self, text: str) -> bool:
        return self._plan_formation_intent(text=text).is_match

    def _decision_intent(self, *, text: str, structured_intent: IntentName | None = None) -> IntentDecision:
        return self._classify_intent_llm(
            intent=IntentName.DECISION_MEMORY_WRITE,
            text=text,
            guideline="Message contains a stable decision worth storing in long-term decisions.",
            structured_intent=structured_intent,
        )

    def _is_decision_intent(self, text: str) -> bool:
        return self._decision_intent(text=text).is_match

    def _note_intent(self, *, text: str, structured_intent: IntentName | None = None) -> IntentDecision:
        return self._classify_intent_llm(
            intent=IntentName.NOTE_MEMORY_WRITE,
            text=text,
            guideline="Message contains an important reusable note/fact worth storing.",
            structured_intent=structured_intent,
        )

    def _is_note_intent(self, text: str) -> bool:
        return self._note_intent(text=text).is_match

    def _classify_intent_llm(
        self,
        *,
        intent: IntentName,
        text: str,
        guideline: str,
        structured_intent: IntentName | None = None,
    ) -> IntentDecision:
        normalized = str(text or "").strip()
        if not normalized:
            return IntentDecision.no_match(
                intent=intent,
                confidence=1.0,
                reason_code="intent_no_match_empty_message",
            )
        structured_name = structured_intent.value if structured_intent else ""
        key = (intent.value, normalized, structured_name)
        if key in self._intent_cache:
            return self._intent_cache[key]
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
            self._intent_cache[key] = result
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
            self._intent_cache[key] = result
            return result
        if not self._reserve_aux_llm_slot(f"router_intent:{intent.value}"):
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
            self._intent_cache[key] = result
            return result

        prompt = (
            "You are an intent classifier for a task-memory system.\n"
            "Return ONLY valid JSON with schema:\n"
            '{"match": true, "confidence": 0.0, "reason": "short"}\n'
            f"Intent: {intent.value}\n"
            f"Guideline: {guideline}\n"
            f"User message: {normalized}\n"
            "Set match=true only if intent is explicit enough."
        )
        try:
            response = self.llm_client.chat_completion(
                model=self.step_parser_model,
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
                result = IntentDecision.unknown(
                    intent=intent,
                    reason_code="intent_unknown_invalid_payload",
                )
                self._intent_cache[key] = result
                logger.info(
                    "[INTENT_CLASSIFY] intent=%s status=%s reason_code=%s",
                    intent.value,
                    result.status,
                    result.reason_code,
                )
                return result
            confidence = self._clamp_confidence(payload.get("confidence"))
            if confidence < 0.6:
                result = IntentDecision.unknown(
                    intent=intent,
                    confidence=confidence,
                    reason=str(payload.get("reason") or ""),
                    reason_code="intent_unknown_low_confidence",
                )
                self._intent_cache[key] = result
                logger.info(
                    "[INTENT_CLASSIFY] intent=%s status=%s reason_code=%s",
                    intent.value,
                    result.status,
                    result.reason_code,
                )
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
            self._intent_cache[key] = result
            return result
        except Exception as exc:
            logger.warning("[INTENT_CLASSIFY] intent=%s failed: %s", intent.value, exc)
            result = IntentDecision.unknown(
                intent=intent,
                reason=str(exc),
                reason_code="intent_unknown_unavailable",
            )
            self._intent_cache[key] = result
            logger.info(
                "[INTENT_CLASSIFY] intent=%s status=%s reason_code=%s",
                intent.value,
                result.status,
                result.reason_code,
            )
            return result

    def _reserve_aux_llm_slot(self, purpose: str) -> bool:
        reserver = self._aux_llm_budget_reserver
        if reserver is None:
            return True
        try:
            return bool(reserver(str(purpose or "")))
        except Exception:
            return False

    def _extract_goal(self, text: str, *, limit: int = 100) -> str:
        normalized = " ".join(str(text or "").split()).strip()
        if not normalized:
            return "Текущая задача"
        return normalized[:limit]

    def _recover_intent_with_separate_analysis(
        self,
        *,
        intent: IntentName,
        text: str,
        guideline: str,
    ) -> IntentDecision | None:
        normalized = str(text or "").strip()
        if not normalized or self.llm_client is None:
            return None
        if not self._reserve_aux_llm_slot(f"router_intent_recovery:{intent.value}"):
            return None

        prompt = (
            "You are a recovery intent classifier.\n"
            "Primary JSON classifier was inconclusive, run a separate analysis pass.\n"
            "Decide whether the message explicitly matches the target intent.\n"
            "Return ONLY valid JSON with schema:\n"
            '{"match": true, "confidence": 0.0, "reason": "short"}\n'
            f"Intent: {intent.value}\n"
            f"Guideline: {guideline}\n"
            f"User message: {normalized}\n"
        )
        try:
            response = self.llm_client.chat_completion(
                model=self.step_parser_model,
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
            confidence = self._clamp_confidence(payload.get("confidence"))
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

    def _build_fallback_plan(self, *, goal: str) -> list[str]:
        goal_text = " ".join(str(goal or "").split()) or "текущей задачи"
        return [
            f"Уточнить требования и критерии готовности для: {goal_text}",
            "Разбить реализацию на минимальные проверяемые шаги",
            "Выполнить шаги и проверить результат тестами/валидацией",
        ]

    @staticmethod
    def _extract_pending_confirmation_id(
        *,
        structured_intent: IntentName | None,
        payload: dict[str, Any],
    ) -> int | None:
        if structured_intent != IntentName.CONFIRM_PENDING_MEMORY:
            return None
        raw_id = payload.get("pending_id")
        try:
            pending_id = int(raw_id)
        except Exception:
            return None
        return pending_id if pending_id > 0 else None

    @staticmethod
    def _extract_first_json_object(text: str) -> dict[str, Any] | None:
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
        for candidate in MemoryRouter._extract_braced_candidates(raw):
            normalized = MemoryRouter._normalize_pythonish_json(candidate)
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
        response: Any,
        required_keys: set[str],
    ) -> tuple[str, dict[str, Any] | None]:
        raw_text = ""
        try:
            raw_text = str(response.choices[0].message.content or "").strip()
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
        value: Any,
        *,
        required_keys: set[str],
        _visited: set[int] | None = None,
    ) -> dict[str, Any] | None:
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
    def _payload_has_required_keys(payload: dict[str, Any] | None, *, required_keys: set[str]) -> bool:
        if not isinstance(payload, dict):
            return False
        keys = {str(k).strip().lower() for k in payload.keys()}
        return required_keys.issubset(keys)

    @staticmethod
    def _safe_dump_raw_response(raw_response: Any) -> str:
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

    @staticmethod
    def _should_auto_start_task_context(*, text: str, structured_intent: IntentName | None) -> bool:
        if structured_intent == IntentName.CONFIRM_PENDING_MEMORY:
            return False
        if structured_intent is not None:
            return True
        return bool(str(text or "").strip())

    def _guard_profile_source(self, source: str) -> None:
        normalized = str(source or "").strip()
        if normalized in self.PROFILE_FORBIDDEN_SOURCES:
            raise ValueError(f"Profile write from forbidden source: {normalized}")

    def _apply_planning_patch(
        self,
        *,
        working: WorkingMemory,
        session_id: str,
        ctx,
        patch: dict[str, Any],
    ) -> list[str]:
        changed_keys: list[str] = []
        plan = list(ctx.plan)
        patch_plan = self._normalize_str_list(patch.get("plan"))
        if patch_plan:
            plan = patch_plan[:10]
            changed_keys.append("plan")
        self._append_unique(plan, patch.get("plan_steps_to_add") or [])
        if plan != ctx.plan:
            changed_keys.append("plan")

        current_step = ctx.current_step
        patch_current_step = str(patch.get("current_step") or "").strip()
        if patch_current_step:
            if patch_current_step not in plan:
                plan.append(patch_current_step)
                if "plan" not in changed_keys:
                    changed_keys.append("plan")
            current_step = patch_current_step
            changed_keys.append("current_step")
        elif not current_step and plan:
            current_step = self._pick_first_pending_step(plan=plan, done_steps=ctx.done)
            changed_keys.append("current_step")

        vars_patch = dict(ctx.vars)
        requirements = list(vars_patch.get("requirements") or [])
        self._append_unique(requirements, patch.get("requirements_to_add") or [])
        if requirements != list(vars_patch.get("requirements") or []):
            vars_patch["requirements"] = requirements
            changed_keys.append("vars")

        artifacts = [a.to_dict() if hasattr(a, "to_dict") else a for a in list(ctx.artifacts)]
        artifact_updates = []
        for ref in patch.get("artifacts_to_add") or []:
            s = str(ref or "").strip()
            if s:
                artifact_updates.append({"step": str(current_step or ""), "type": ArtifactType.RESPONSE.value, "ref": s})
        if artifact_updates:
            artifacts.extend(artifact_updates)
            changed_keys.append("artifacts")

        if changed_keys:
            working.update(
                session_id,
                plan=plan,
                current_step=current_step,
                vars=vars_patch,
                artifacts=artifacts,
            )
        return list(dict.fromkeys(changed_keys))

    def _apply_execution_patch(
        self,
        *,
        working: WorkingMemory,
        session_id: str,
        ctx,
        patch: dict[str, Any],
    ) -> list[str]:
        changed_keys: list[str] = []
        if patch.get("plan_steps_to_add") or patch.get("current_step"):
            logger.info(
                "[MEMORY_WRITE] layer=working blocked=true state=EXECUTION reason=plan_or_current_step_mutation_forbidden"
            )

        completion_intent = self._is_step_completion_intent(
            current_step=ctx.current_step,
            done_steps_to_add=patch.get("done_steps_to_add") or [],
        )

        artifact_refs = [str(x).strip() for x in (patch.get("artifacts_to_add") or []) if str(x).strip()]
        if completion_intent:
            artifact_payload: dict[str, str] | None = None
            if artifact_refs:
                artifact_payload = {
                    "step": str(ctx.current_step or ""),
                    "type": ArtifactType.RESPONSE.value,
                    "ref": artifact_refs[0],
                }
            updated_ctx = working.complete_current_step(session_id=session_id, artifact=artifact_payload)
            changed_keys.extend(["done", "current_step"])
            if artifact_payload:
                changed_keys.append("artifacts")
            if updated_ctx.current_step is None and updated_ctx.done == updated_ctx.plan:
                working.request_validation(session_id)
                changed_keys.append("state")
                logger.info(
                    "[STATE_AUTO] EXECUTION -> VALIDATION (all plan steps completed)"
                )
            return list(dict.fromkeys(changed_keys))

        if artifact_refs:
            for ref in artifact_refs:
                working.append_artifact_for_current_step(
                    session_id=session_id,
                    artifact={"step": str(ctx.current_step or ""), "type": ArtifactType.RESPONSE.value, "ref": ref},
                )
            changed_keys.append("artifacts")
            return changed_keys

        return changed_keys

    def _is_step_completion_intent(
        self,
        *,
        current_step: str | None,
        done_steps_to_add: list[str],
    ) -> bool:
        normalized_done = [str(x).strip() for x in done_steps_to_add if str(x).strip()]
        if current_step and normalized_done and normalized_done[0] == str(current_step):
            return True
        return False
