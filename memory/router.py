from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from typing import TYPE_CHECKING, Any

from memory.long_term import LongTermMemory
from memory.models import ArtifactType, MemoryWriteEvent, TaskState
from memory.working import WorkingMemory

if TYPE_CHECKING:
    from llm.client import LLMClient

logger = logging.getLogger("memory")


class MemoryRouter:
    WORKING_CONFIDENCE_THRESHOLD = 0.65
    PROFILE_FORBIDDEN_SOURCES = ["working_memory", "task_artifact", "step_result"]
    VALIDATION_CONFIRM_PATTERNS = [
        "подтверждаю завершение",
        "всё готово",
        "задача выполнена",
        "переходим к итогам",
        "confirm",
    ]
    VALIDATION_REJECT_PATTERNS = [
        "вернуться к выполнению",
        "нужно доработать",
        "есть замечания",
        "back to execution",
    ]
    DECISION_PATTERNS = [r"\bвыбираем\b", r"\bрешили\b", r"\bиспользуем\b", r"\bстандарт\b", r"\bдоговорились\b"]
    WORKING_PATTERNS = [r"\bтребован", r"\bэндпоинт", r"добавь шаг", r"сделаем так", r"текущий шаг", r"план"]
    NOTE_PATTERNS = [r"\bзапомни\b", r"\bважно\b", r"\bучти\b", r"\bфакт\b"]
    PLAN_FORMATION_PATTERNS = [
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
    TASK_INTENT_PATTERNS = [
        r"\bсостав(?:ь|ьте)?\b.*\bплан\b",
        r"\bсделай\b.*\bанализ\b",
        r"\bпомоги\b.*\bразобраться\b",
        r"\bкак мне\b",
        r"\bчто делать\b",
        r"\bоптимизируй\b",
        r"\bплан на\b",
        r"\bхочу\b.*\b(?:ios|айос|приложени|app)\b",
        r"\bдавай\b.*\b(?:созда(?:ть|дим)|сдела(?:ть|ем)|напиш(?:ем|и))\b.*\b(?:ios|приложени|app)\b",
        r"\bс нуля\b.*\b(?:ios|приложени|app)\b",
    ]
    EXECUTION_ALLOW_PATTERNS = [
        r"\bполный\b.*\bплан\b",
        r"\bнужен\b.*\bплан\b",
        r"\bс нуля\b",
        r"\bдо первой покупки\b",
        r"\bархитектур",
        r"\bmvp\b",
        r"\broadmap\b",
        r"\bмонетизац",
        r"\bstorekit\b",
        r"\bswiftui\b",
        r"\bчеклист\b",
        r"\bследующ(?:ий|ие)\s+шаг",
        r"\bкак\b.*\bсделать\b",
        r"\bкак\b.*\bреализовать\b",
    ]
    EXECUTION_DENY_PATTERNS = [
        r"\bзабудь\b",
        r"\bдругая задача\b",
        r"\bдругой проект\b",
        r"\bсменим тему\b",
        r"\bсмена контекста\b",
        r"\bпереключим(?:ся)?\b.*\bзадач",
    ]

    def __init__(self, *, llm_client: "LLMClient | None" = None, step_parser_model: str = "gpt-5-nano"):
        self.llm_client = llm_client
        self.step_parser_model = str(step_parser_model or "gpt-5-nano")

    def route_user_message(
        self,
        *,
        session_id: str,
        user_id: str,
        user_message: str,
        working: WorkingMemory,
        long_term: LongTermMemory,
    ) -> list[MemoryWriteEvent]:
        text = (user_message or "").strip()
        lower = text.lower()
        events: list[MemoryWriteEvent] = []

        confirm_match = re.search(r"подтверждаю память\s*#\s*(\d+)", lower, flags=re.IGNORECASE)
        if confirm_match:
            pending_id = int(confirm_match.group(1))
            self._guard_profile_source("explicit_confirmation")
            approved = long_term.approve_pending_entry(user_id=user_id, pending_id=pending_id)
            if approved:
                events.append(MemoryWriteEvent(layer="long_term.approval", keys=["pending_id", "type"]))
            else:
                logger.info("[MEMORY_WRITE] layer=long_term.approval status=not_found pending_id=%s", pending_id)
            return events

        if self._matches_any(lower, self.DECISION_PATTERNS):
            tags = ["decision"]
            if "стандарт" in lower:
                tags.append("standard")
            result = long_term.add_decision(user_id=user_id, text=text, tags=tags, source="user")
            if result.get("status") == "pending":
                events.append(MemoryWriteEvent(layer="long_term.pending", keys=["pending_id"]))
            else:
                events.append(MemoryWriteEvent(layer="long_term.decision", keys=["text", "tags"]))

        if self._matches_any(lower, self.NOTE_PATTERNS):
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
        plan_formation_intent = self._is_plan_formation_intent(lower)
        task_intent = self._is_task_intent(lower)
        if existing_ctx is None and (task_intent or plan_formation_intent):
            auto_goal = self._extract_goal(text)
            existing_ctx = working.start_task(session_id=session_id, goal=auto_goal)
            events.append(MemoryWriteEvent(layer="working", keys=["task", "state"]))
            logger.info('[TASK_AUTO_START] goal="%s" session=%s', auto_goal, session_id)
        existing_state = existing_ctx.state if existing_ctx else TaskState.PLANNING
        regex_patch = self._extract_working_patch_from_regex(
            text=text,
            lower=lower,
            existing_current_step=existing_ctx.current_step if existing_ctx else None,
        )
        llm_patch = self._extract_working_patch_via_llm(
            text=text,
            current_plan=list(existing_ctx.plan) if existing_ctx else [],
            current_step=existing_ctx.current_step if existing_ctx else None,
            done_steps=list(existing_ctx.done) if existing_ctx else [],
            working_state=(existing_ctx.state.value if existing_ctx else "NONE"),
            goal=(existing_ctx.task if existing_ctx else self._extract_goal(text)),
        )
        llm_applied = False
        llm_confidence = float(llm_patch.get("confidence", 0.0)) if llm_patch else 0.0
        llm_keys = self._working_patch_keys(llm_patch) if llm_patch else []
        if (
            llm_patch
            and bool(llm_patch.get("is_working_update"))
            and llm_confidence >= self.WORKING_CONFIDENCE_THRESHOLD
            and self._working_patch_has_changes(llm_patch)
        ):
            regex_patch = self._merge_working_patches(regex_patch, llm_patch)
            llm_applied = True
        if plan_formation_intent and llm_confidence == 0.0:
            logger.info('[WORKING_EXTRACT_FALLBACK] message="%s" reason="zero_confidence_plan_formation"', text)
            if existing_ctx is None:
                fallback_goal = self._extract_goal(text)
                existing_ctx = working.start_task(session_id=session_id, goal=fallback_goal)
                events.append(MemoryWriteEvent(layer="working", keys=["task", "state"]))
                logger.info('[TASK_AUTO_START] goal="%s" session=%s', fallback_goal, session_id)
            fallback_plan = self._build_fallback_plan(
                goal=str(existing_ctx.task or existing_ctx.goal or self._extract_goal(text))
            )
            if fallback_plan:
                regex_patch["plan"] = fallback_plan
                regex_patch["current_step"] = fallback_plan[0]
                regex_patch["is_working_update"] = True
                llm_applied = True
                llm_confidence = 0.7
                llm_keys = list(dict.fromkeys(list(llm_keys) + ["plan", "current_step"]))
                logger.info(
                    '[WORKING_EXTRACT_FALLBACK] action="autoplan_built" steps=%s session=%s',
                    len(fallback_plan),
                    session_id,
                )
            else:
                vars_patch = dict(existing_ctx.vars)
                if not vars_patch.get("plan_guidance_required"):
                    vars_patch["plan_guidance_required"] = True
                    working.update(session_id, vars=vars_patch)
                    events.append(MemoryWriteEvent(layer="working", keys=["vars"]))
        elif task_intent and llm_confidence == 0.0:
            logger.info('[WORKING_EXTRACT_FALLBACK] message="%s" reason="zero_confidence"', text)
            if existing_ctx is None:
                fallback_goal = self._extract_goal(text)
                existing_ctx = working.start_task(session_id=session_id, goal=fallback_goal)
                events.append(MemoryWriteEvent(layer="working", keys=["task", "state"]))
                logger.info('[TASK_AUTO_START] goal="%s" session=%s', fallback_goal, session_id)
            llm_applied = True
            llm_confidence = 0.85
            if "task" not in llm_keys:
                llm_keys = list(llm_keys) + ["task"]
        logger.info(
            "[MEMORY_WORKING_EXTRACT] source=llm applied=%s confidence=%.2f keys=%s",
            llm_applied,
            llm_confidence,
            ",".join(llm_keys) if llm_keys else "-",
        )

        if self._working_patch_has_changes(regex_patch):
            ctx = existing_ctx or working.ensure_task(session_id=session_id, goal="Текущая задача")
            state = ctx.state
            try:
                if state == TaskState.PLANNING:
                    changed_keys = self._apply_planning_patch(
                        working=working,
                        session_id=session_id,
                        ctx=ctx,
                        patch=regex_patch,
                    )
                    if plan_formation_intent:
                        updated_ctx = working.load(session_id)
                        if (
                            updated_ctx
                            and updated_ctx.state == TaskState.PLANNING
                            and bool(updated_ctx.plan)
                            and updated_ctx.current_step == updated_ctx.plan[0]
                        ):
                            try:
                                working.transition_state(updated_ctx, TaskState.EXECUTION)
                                updated_ctx.updated_at = datetime.utcnow().isoformat()
                                working.save(updated_ctx)
                                changed_keys.append("state")
                                logger.info("[STATE_AUTO] PLANNING -> EXECUTION (plan formed)")
                            except ValueError as exc:
                                logger.info(
                                    "[MEMORY_WRITE] layer=working blocked=true state=PLANNING reason=auto_transition_failed:%s",
                                    str(exc),
                                )
                    if changed_keys:
                        events.append(MemoryWriteEvent(layer="working", keys=changed_keys))
                elif state == TaskState.EXECUTION:
                    changed_keys = self._apply_execution_patch(
                        working=working,
                        session_id=session_id,
                        ctx=ctx,
                        text=text,
                        lower=lower,
                        patch=regex_patch,
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
        lower = normalized.lower()
        if self._matches_any(lower, self.EXECUTION_DENY_PATTERNS):
            return False
        if self._matches_any(lower, self.EXECUTION_ALLOW_PATTERNS):
            return True
        if self.llm_client is None:
            return self._fallback_execution_allowance(text=normalized, current_step=step)

        prompt = f"""You classify whether a user message is allowed in EXECUTION state of a task-state machine.
Return ONLY valid JSON.

Schema:
{{
  "allow": true/false,
  "confidence": 0.0,
  "reason": "short"
}}

Rules:
- allow=true when the message is about the active step progress, implementation details, code requests, clarification for current solution, or marking step completion.
- allow=false when the message tries to switch to another task/topic, asks to skip process control, or requests final full result that bypasses current step flow.
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
            )
            raw = str(response.choices[0].message.content or "").strip()
            match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
            if not match:
                return self._fallback_execution_allowance(text=normalized, current_step=step)
            payload = json.loads(match.group())
            if not isinstance(payload, dict):
                return self._fallback_execution_allowance(text=normalized, current_step=step)
            allow = bool(payload.get("allow"))
            confidence = self._clamp_confidence(payload.get("confidence"))
            if confidence < 0.5:
                return self._fallback_execution_allowance(text=normalized, current_step=step)
            return allow
        except Exception as exc:
            logger.warning("[EXECUTION_GUARD] llm classify failed: %s", exc)
            return self._fallback_execution_allowance(text=normalized, current_step=step)

    def _extract_working_patch_from_regex(
        self,
        *,
        text: str,
        lower: str,
        existing_current_step: str | None,
    ) -> dict[str, Any]:
        patch = self._empty_working_patch()

        if "требован" in lower or "эндпоинт" in lower:
            patch["requirements_to_add"] = [text]

        step_match = re.search(r"добавь шаг[:\s]+(.+)$", text, flags=re.IGNORECASE)
        if step_match:
            step = step_match.group(1).strip()
            if step:
                patch["plan_steps_to_add"] = [step]
                if not patch["current_step"] and not str(existing_current_step or "").strip():
                    patch["current_step"] = step

        current_step_match = re.search(r"текущий шаг[:\s]+(.+)$", text, flags=re.IGNORECASE)
        if current_step_match:
            patch["current_step"] = current_step_match.group(1).strip()

        done_match = re.search(r"(?:шаг выполнен|выполнил(?:\s+шаг)?|done step)[:\s]+(.+)$", text, flags=re.IGNORECASE)
        if done_match:
            step = done_match.group(1).strip()
            if step:
                patch["done_steps_to_add"] = [step]
        elif re.search(r"\b(шаг выполнен|готово|выполнено|done|completed)\b", lower, flags=re.IGNORECASE):
            if existing_current_step:
                patch["done_steps_to_add"] = [str(existing_current_step)]

        if "артефакт" in lower:
            patch["artifacts_to_add"] = [text]

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
    ) -> dict[str, Any]:
        if not text or self.llm_client is None:
            return {}

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
            )
            raw = str(response.choices[0].message.content or "").strip()
            logger.info("[MEMORY_WORKING_EXTRACT_RAW] response=%s", raw)
            match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
            if not match:
                return {}
            payload = json.loads(match.group())
            if not isinstance(payload, dict):
                return {}
            return self._normalize_working_patch_payload(payload)
        except Exception as exc:
            logger.warning("[MEMORY_WORKING_EXTRACT] source=llm parse_error=%s", exc)
            return {}

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

    def _merge_working_patches(self, base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
        merged = self._normalize_working_patch_payload(base)
        extra_norm = self._normalize_working_patch_payload(extra)

        for key in ["plan_steps_to_add", "done_steps_to_add", "requirements_to_add", "artifacts_to_add"]:
            self._append_unique(merged[key], extra_norm[key])

        extra_plan = self._normalize_str_list(extra_norm.get("plan"))
        if extra_plan:
            merged["plan"] = extra_plan[:10]

        extra_task = str(extra_norm.get("task") or "").strip()
        if extra_task:
            merged["task"] = extra_task

        extra_current_step = str(extra_norm.get("current_step") or "").strip()
        if extra_current_step:
            merged["current_step"] = extra_current_step

        merged["confidence"] = max(
            float(merged.get("confidence", 0.0)),
            float(extra_norm.get("confidence", 0.0)),
        )
        merged["is_working_update"] = bool(
            merged.get("is_working_update")
            or extra_norm.get("is_working_update")
            or self._working_patch_has_changes(merged)
        )
        return merged

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

    def _matches_any(self, text: str, patterns: list[str]) -> bool:
        return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)

    def _is_task_intent(self, text: str) -> bool:
        return self._matches_any(text, self.TASK_INTENT_PATTERNS)

    def _is_plan_formation_intent(self, text: str) -> bool:
        return self._matches_any(text, self.PLAN_FORMATION_PATTERNS)

    def _extract_goal(self, text: str, *, limit: int = 100) -> str:
        normalized = re.sub(r"\s+", " ", str(text or "")).strip()
        if not normalized:
            return "Текущая задача"
        return normalized[:limit]

    def _build_fallback_plan(self, *, goal: str) -> list[str]:
        goal_text = re.sub(r"\s+", " ", str(goal or "")).strip() or "текущей задачи"
        return [
            f"Уточнить требования и критерии готовности для: {goal_text}",
            "Разбить реализацию на минимальные проверяемые шаги",
            "Выполнить шаги и проверить результат тестами/валидацией",
        ]

    @staticmethod
    def _fallback_execution_allowance(*, text: str, current_step: str) -> bool:
        lower = str(text or "").lower()
        step_lower = str(current_step or "").strip().lower()
        if not lower:
            return False
        context_switch_markers = (
            "забудь",
            "другая задача",
            "другой проект",
            "сменим тему",
            "смена контекста",
        )
        if any(marker in lower for marker in context_switch_markers):
            return False
        helpful_request_markers = (
            "полный план",
            "нужен план",
            "до первой покупки",
            "архитектур",
            "swiftui",
            "mvp",
            "монетизац",
            "чеклист",
            "следующий шаг",
        )
        if any(marker in lower for marker in helpful_request_markers):
            return True
        completion_markers = ("шаг выполнен", "выполнил", "выполнено", "готово", "done", "completed")
        if step_lower and any(marker in lower for marker in completion_markers):
            return True
        if step_lower and step_lower in lower:
            return True
        return False

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
        text: str,
        lower: str,
        patch: dict[str, Any],
    ) -> list[str]:
        del text
        changed_keys: list[str] = []
        if patch.get("plan_steps_to_add") or patch.get("current_step"):
            logger.info(
                "[MEMORY_WRITE] layer=working blocked=true state=EXECUTION reason=plan_or_current_step_mutation_forbidden"
            )

        completion_intent = self._is_step_completion_intent(
            lower=lower,
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
        lower: str,
        current_step: str | None,
        done_steps_to_add: list[str],
    ) -> bool:
        normalized_done = [str(x).strip() for x in done_steps_to_add if str(x).strip()]
        if current_step and normalized_done and normalized_done[0] == str(current_step):
            return True
        completion_markers = [
            "шаг выполнен",
            "выполнил шаг",
            "выполнено",
            "готово",
            "completed",
            "done",
            "завершил шаг",
        ]
        return bool(current_step and any(marker in lower for marker in completion_markers))
