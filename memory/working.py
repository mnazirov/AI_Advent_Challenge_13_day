from __future__ import annotations

import logging
from datetime import datetime
from uuid import uuid4

import storage
from memory.models import ArtifactType, TaskArtifact, TaskContext, TaskState

logger = logging.getLogger("memory")


class WorkingMemory:
    ALLOWED_TRANSITIONS: dict[TaskState, set[TaskState]] = {
        TaskState.PLANNING: {TaskState.EXECUTION},
        TaskState.EXECUTION: {TaskState.VALIDATION, TaskState.PLANNING},
        TaskState.VALIDATION: {TaskState.DONE, TaskState.EXECUTION},
        TaskState.DONE: set(),
    }

    def start_task(self, session_id: str, goal: str) -> TaskContext:
        ctx = TaskContext(
            session_id=session_id,
            task_id=f"task-{uuid4().hex[:8]}",
            task=str(goal or "").strip(),
            state=TaskState.PLANNING,
            updated_at=datetime.utcnow().isoformat(),
        )
        self._validate_context(ctx)
        self.save(ctx)
        return ctx

    def load(self, session_id: str) -> TaskContext | None:
        payload = storage.memory_load_working_task(session_id=session_id)
        if not payload:
            return None
        ctx = TaskContext.from_dict(payload)
        self._validate_context(ctx)
        repaired, chosen_step = self._repair_legacy_current_step(ctx)
        if repaired:
            logger.info(
                "[MEMORY_WORKING_REPAIR] repaired=%s session_id=%s chosen_step=%s",
                True,
                ctx.session_id,
                chosen_step,
            )
            self.save(ctx)
        return ctx

    def save(self, ctx: TaskContext) -> None:
        self._validate_context(ctx)
        payload = ctx.to_dict()
        storage.memory_save_working_task(
            session_id=payload["session_id"],
            task_id=payload["task_id"],
            task=payload["task"],
            goal=payload["goal"],
            state=payload["state"],
            plan=payload["plan"],
            current_step=payload["current_step"] or "",
            done=payload["done"],
            done_steps=payload["done_steps"],
            open_questions=payload["open_questions"],
            artifacts=payload["artifacts"],
            vars_data=payload["vars"],
            updated_at=payload["updated_at"],
        )

    def clear_session(self, session_id: str) -> None:
        storage.memory_clear_working_task(session_id=session_id)

    def update(self, session_id: str, **patch: object) -> TaskContext:
        ctx = self.load(session_id)
        if ctx is None:
            goal = str(patch.get("task") or patch.get("goal") or "Текущая задача")
            ctx = self.start_task(session_id=session_id, goal=goal or "Текущая задача")

        self._ensure_not_done_for_write(ctx)

        if "goal" in patch or "task" in patch:
            candidate = str(patch.get("task") or patch.get("goal") or "").strip()
            if candidate and candidate != ctx.task:
                raise ValueError("task is immutable after start")

        if "plan" in patch:
            if ctx.state != TaskState.PLANNING:
                raise ValueError("plan mutation is only allowed in PLANNING state")
            ctx.plan = self._normalize_step_list(patch.get("plan"))

        if "current_step" in patch:
            raw_step = patch.get("current_step")
            next_step = str(raw_step).strip() if raw_step is not None else ""
            ctx.current_step = next_step or None

        done_patch: list[str] | None = None
        if "done" in patch:
            done_patch = self._normalize_step_list(patch.get("done"))
        if "done_steps" in patch:
            done_patch = self._normalize_step_list(patch.get("done_steps"))
        if done_patch is not None:
            if ctx.state in {TaskState.EXECUTION, TaskState.VALIDATION} and done_patch != ctx.done:
                raise ValueError("done can only be mutated via complete_current_step()")
            ctx.done = done_patch

        if "open_questions" in patch and isinstance(patch["open_questions"], list):
            ctx.open_questions = [str(x).strip() for x in patch["open_questions"] if str(x).strip()]

        if "artifacts" in patch and isinstance(patch["artifacts"], list):
            normalized_artifacts: list[TaskArtifact] = []
            for item in patch["artifacts"]:
                artifact = TaskArtifact.from_any(item)
                if artifact.ref:
                    normalized_artifacts.append(artifact)
            ctx.artifacts = normalized_artifacts

        if "vars" in patch and isinstance(patch["vars"], dict):
            merged = dict(ctx.vars)
            merged.update(patch["vars"])
            ctx.vars = merged

        if "state" in patch and patch["state"]:
            self.transition_state(ctx, TaskState(str(patch["state"])))

        self._validate_context(ctx)
        ctx.updated_at = datetime.utcnow().isoformat()
        self.save(ctx)
        return ctx

    def transition_state(self, ctx: TaskContext, new_state: TaskState) -> TaskContext:
        old = ctx.state
        if old == new_state:
            return ctx

        if new_state not in self.ALLOWED_TRANSITIONS.get(old, set()):
            raise ValueError(f"Transition {old.value} -> {new_state.value} is forbidden")

        if old == TaskState.PLANNING and new_state == TaskState.EXECUTION:
            if not ctx.plan:
                raise ValueError("plan is empty")
            if ctx.current_step != ctx.plan[0]:
                raise ValueError("current_step must equal plan[0]")

        if old == TaskState.EXECUTION and new_state == TaskState.VALIDATION:
            if ctx.done != ctx.plan:
                raise ValueError("done != plan: not all steps completed in order")

        if old == TaskState.EXECUTION and new_state == TaskState.PLANNING:
            ctx.current_step = None
            ctx.done = []
            logger.info(
                "[ROLLBACK] EXECUTION -> PLANNING | cleared: current_step, done | preserved: artifacts, open_questions"
            )

        if old == TaskState.VALIDATION and new_state == TaskState.EXECUTION:
            if ctx.done:
                # Reopen the last completed step for rework.
                ctx.current_step = ctx.done.pop()
            elif ctx.plan:
                ctx.current_step = ctx.plan[0]
            else:
                ctx.current_step = None
            logger.info(
                '[ROLLBACK] VALIDATION -> EXECUTION | restored: current_step="%s" | preserved: artifacts, open_questions',
                ctx.current_step or "",
            )

        if old == TaskState.VALIDATION and new_state == TaskState.DONE:
            ctx.current_step = None

        ctx.state = new_state
        if old == TaskState.VALIDATION and new_state == TaskState.DONE and ctx.open_questions:
            logger.warning("[WARN] DONE reached with open_questions: %s", ctx.open_questions)

        status = self.get_step_status(ctx)
        logger.info(
            '[STATE] %s -> %s | step %s/%s: "%s"',
            old.value,
            new_state.value,
            status.get("step_index") if status.get("step_index") is not None else len(ctx.done),
            status.get("total_steps", 0),
            ctx.current_step or "",
        )
        self._validate_context(ctx)
        return ctx

    def ensure_task(self, session_id: str, goal: str = "") -> TaskContext:
        current = self.load(session_id)
        if current is not None:
            return current
        return self.start_task(session_id=session_id, goal=goal or "Текущая задача")

    def complete_current_step(self, session_id: str, artifact: TaskArtifact | dict | str | None = None) -> TaskContext:
        ctx = self.load(session_id)
        if ctx is None:
            raise ValueError("working task not found")
        self._ensure_not_done_for_write(ctx)
        if ctx.state != TaskState.EXECUTION:
            raise ValueError("complete_current_step is only allowed in EXECUTION state")

        current_step = str(ctx.current_step or "").strip()
        if not current_step or current_step not in ctx.plan:
            raise ValueError(f"current_step '{current_step}' not found in plan")
        if current_step in ctx.done:
            raise ValueError(f"step '{current_step}' already completed")
        expected_index = len(ctx.done)
        if expected_index >= len(ctx.plan) or ctx.plan[expected_index] != current_step:
            raise ValueError(f"cannot mark done: '{current_step}' is not current_step")

        ctx.done.append(current_step)
        if artifact is not None:
            parsed = TaskArtifact.from_any(artifact)
            ref = str(parsed.ref or "").strip()
            if ref:
                art_type = parsed.type
                if art_type == ArtifactType.LEGACY:
                    art_type = ArtifactType.RESPONSE
                ctx.artifacts.append(TaskArtifact(step=current_step, type=art_type, ref=ref))

        if len(ctx.done) < len(ctx.plan):
            ctx.current_step = ctx.plan[len(ctx.done)]
        else:
            ctx.current_step = None

        ctx.updated_at = datetime.utcnow().isoformat()
        self.save(ctx)
        logger.info(
            '[STEP] %s/%s done: "%s" -> current: "%s"',
            len(ctx.done),
            len(ctx.plan),
            current_step,
            ctx.current_step or "",
        )
        return ctx

    def append_artifact_for_current_step(self, session_id: str, artifact: TaskArtifact | dict | str) -> TaskContext:
        ctx = self.load(session_id)
        if ctx is None:
            raise ValueError("working task not found")
        self._ensure_not_done_for_write(ctx)
        if ctx.state != TaskState.EXECUTION:
            raise ValueError("artifact append is only allowed in EXECUTION state")

        current_step = str(ctx.current_step or "").strip()
        if not current_step:
            raise ValueError("no active current_step to attach artifact")

        parsed = TaskArtifact.from_any(artifact)
        ref = str(parsed.ref or "").strip()
        if not ref:
            raise ValueError("artifact ref is empty")
        art_type = parsed.type if parsed.type != ArtifactType.LEGACY else ArtifactType.RESPONSE
        ctx.artifacts.append(TaskArtifact(step=current_step, type=art_type, ref=ref))
        ctx.updated_at = datetime.utcnow().isoformat()
        self.save(ctx)
        return ctx

    def request_validation(self, session_id: str) -> TaskContext:
        ctx = self.load(session_id)
        if ctx is None:
            raise ValueError("working task not found")
        self._ensure_not_done_for_write(ctx)
        if ctx.state != TaskState.EXECUTION:
            raise ValueError("request_validation is only allowed in EXECUTION state")
        if ctx.done != ctx.plan:
            raise ValueError("done != plan")
        self.transition_state(ctx, TaskState.VALIDATION)
        ctx.updated_at = datetime.utcnow().isoformat()
        self.save(ctx)
        return ctx

    def get_step_status(self, ctx: TaskContext) -> dict[str, object]:
        return {
            "state": ctx.state.value,
            "current_step": ctx.current_step,
            "step_index": len(ctx.done) + 1 if ctx.state == TaskState.EXECUTION else None,
            "total_steps": len(ctx.plan),
        }

    def _repair_legacy_current_step(self, ctx: TaskContext) -> tuple[bool, str]:
        if ctx.state != TaskState.EXECUTION:
            return False, ""
        if ctx.current_step or not ctx.plan:
            return False, ""
        if ctx.done == ctx.plan:
            return False, ""
        chosen_step = self._select_current_step(plan=ctx.plan, done_steps=ctx.done)
        if not chosen_step:
            return False, ""
        ctx.current_step = chosen_step
        ctx.updated_at = datetime.utcnow().isoformat()
        return True, chosen_step

    @staticmethod
    def _select_current_step(*, plan: list[str], done_steps: list[str]) -> str:
        done_set = {str(x) for x in (done_steps or [])}
        for step in plan:
            normalized = str(step)
            if normalized not in done_set:
                return normalized
        return str(plan[0]) if plan else ""

    def _validate_context(self, ctx: TaskContext) -> None:
        duplicate_plan = self._find_duplicate(ctx.plan)
        if duplicate_plan:
            raise ValueError(f"duplicate step in plan: '{duplicate_plan}'")

        duplicate_done = self._find_duplicate(ctx.done)
        if duplicate_done:
            raise ValueError(f"duplicate entry in done: '{duplicate_done}'")

        for done_step in ctx.done:
            if done_step not in ctx.plan:
                raise ValueError(f"done step '{done_step}' not found in plan")
        if ctx.done != ctx.plan[: len(ctx.done)]:
            raise ValueError("done steps are out of plan order")

        if ctx.current_step and ctx.current_step not in ctx.plan:
            raise ValueError(f"current_step '{ctx.current_step}' not found in plan")

        if ctx.state == TaskState.EXECUTION and ctx.plan and ctx.current_step is None and ctx.done != ctx.plan:
            raise ValueError("current_step must be set in EXECUTION before all steps are done")

        if ctx.state == TaskState.DONE:
            ctx.current_step = None

        if ctx.state in {TaskState.PLANNING, TaskState.DONE} and ctx.current_step == "":
            ctx.current_step = None

    def _ensure_not_done_for_write(self, ctx: TaskContext) -> None:
        if ctx.state == TaskState.DONE:
            raise ValueError("Working memory is frozen in DONE state")

    @staticmethod
    def _normalize_step_list(value: object) -> list[str]:
        if not isinstance(value, list):
            return []
        out: list[str] = []
        for item in value:
            step = str(item or "").strip()
            if step:
                out.append(step)
        return out

    @staticmethod
    def _find_duplicate(values: list[str]) -> str | None:
        seen: set[str] = set()
        for item in values:
            if item in seen:
                return item
            seen.add(item)
        return None
