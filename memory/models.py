from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class TaskState(str, Enum):
    PLANNING = "PLANNING"
    EXECUTION = "EXECUTION"
    VALIDATION = "VALIDATION"
    DONE = "DONE"


class ArtifactType(str, Enum):
    FILE = "file"
    COMMIT = "commit"
    RESPONSE = "response"
    LEGACY = "legacy"


class ProfileSource(str, Enum):
    USER_EXPLICIT = "user_explicit"
    AGENT_INFERRED = "agent_inferred"
    DEBUG_MENU = "debug_menu"


def _now_iso() -> str:
    return datetime.utcnow().isoformat()


@dataclass
class ProfileField:
    value: object
    source: ProfileSource
    verified: bool
    confidence: float | None
    updated_at: str

    def to_dict(self) -> dict[str, Any]:
        value = self.value
        if hasattr(value, "to_dict"):
            try:
                value = value.to_dict()
            except Exception:
                value = self.value
        return {
            "value": value,
            "source": self.source.value,
            "verified": bool(self.verified),
            "confidence": self.confidence if self.confidence is None else float(self.confidence),
            "updated_at": str(self.updated_at or ""),
        }

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, Any],
        *,
        default_value: object,
        default_source: ProfileSource = ProfileSource.USER_EXPLICIT,
        default_verified: bool = True,
    ) -> "ProfileField":
        if not isinstance(payload, dict):
            payload = {}
        source_raw = str(payload.get("source") or default_source.value)
        try:
            source = ProfileSource(source_raw)
        except ValueError:
            source = default_source
        confidence_raw = payload.get("confidence")
        confidence: float | None
        if confidence_raw in (None, ""):
            confidence = None
        else:
            try:
                confidence = float(confidence_raw)
            except Exception:
                confidence = None
        return cls(
            value=payload.get("value", default_value),
            source=source,
            verified=bool(payload.get("verified", default_verified)),
            confidence=confidence,
            updated_at=str(payload.get("updated_at") or ""),
        )


@dataclass
class ProjectContext:
    project_name: str
    goals: list[str]
    key_decisions: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_name": str(self.project_name or ""),
            "goals": [str(x) for x in (self.goals or [])],
            "key_decisions": [str(x) for x in (self.key_decisions or [])],
        }

    @classmethod
    def from_any(cls, value: object) -> "ProjectContext":
        if isinstance(value, ProjectContext):
            return cls(
                project_name=str(value.project_name or ""),
                goals=[str(x) for x in (value.goals or [])],
                key_decisions=[str(x) for x in (value.key_decisions or [])],
            )
        if isinstance(value, dict):
            return cls(
                project_name=str(value.get("project_name") or ""),
                goals=[str(x) for x in (value.get("goals") or [])],
                key_decisions=[str(x) for x in (value.get("key_decisions") or [])],
            )
        if isinstance(value, list):
            # Legacy compatibility: context_json list -> key_decisions
            return cls(project_name="", goals=[], key_decisions=[str(x) for x in value])
        return cls(project_name="", goals=[], key_decisions=[])


@dataclass
class ProfileConflict:
    field: str
    existing_value: object
    inferred_value: object
    confidence: float
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "field": str(self.field or ""),
            "existing_value": self.existing_value,
            "inferred_value": self.inferred_value,
            "confidence": float(self.confidence),
            "created_at": str(self.created_at or ""),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ProfileConflict":
        if not isinstance(payload, dict):
            payload = {}
        confidence_raw = payload.get("confidence")
        try:
            confidence = float(confidence_raw)
        except Exception:
            confidence = 0.0
        return cls(
            field=str(payload.get("field") or ""),
            existing_value=payload.get("existing_value"),
            inferred_value=payload.get("inferred_value"),
            confidence=confidence,
            created_at=str(payload.get("created_at") or ""),
        )


@dataclass
class LongTermProfile:
    stack_tools: ProfileField
    response_style: ProfileField
    hard_constraints: ProfileField
    user_role_level: ProfileField
    project_context: ProfileField
    extra_fields: dict[str, ProfileField]
    conflicts: list[ProfileConflict]

    @classmethod
    def default(cls) -> "LongTermProfile":
        return cls(
            stack_tools=ProfileField(
                value=[],
                source=ProfileSource.USER_EXPLICIT,
                verified=True,
                confidence=None,
                updated_at="",
            ),
            response_style=ProfileField(
                value="",
                source=ProfileSource.USER_EXPLICIT,
                verified=True,
                confidence=None,
                updated_at="",
            ),
            hard_constraints=ProfileField(
                value=[],
                source=ProfileSource.USER_EXPLICIT,
                verified=True,
                confidence=None,
                updated_at="",
            ),
            user_role_level=ProfileField(
                value="",
                source=ProfileSource.USER_EXPLICIT,
                verified=True,
                confidence=None,
                updated_at="",
            ),
            project_context=ProfileField(
                value=ProjectContext(project_name="", goals=[], key_decisions=[]),
                source=ProfileSource.USER_EXPLICIT,
                verified=True,
                confidence=None,
                updated_at="",
            ),
            extra_fields={},
            conflicts=[],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "stack_tools": self.stack_tools.to_dict(),
            "response_style": self.response_style.to_dict(),
            "hard_constraints": self.hard_constraints.to_dict(),
            "user_role_level": self.user_role_level.to_dict(),
            "project_context": self.project_context.to_dict(),
            "extra_fields": {k: v.to_dict() for k, v in (self.extra_fields or {}).items()},
            "conflicts": [c.to_dict() for c in (self.conflicts or [])],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LongTermProfile":
        base = cls.default()
        if not isinstance(payload, dict):
            payload = {}

        stack_tools = ProfileField.from_dict(payload.get("stack_tools") or {}, default_value=[])
        response_style = ProfileField.from_dict(payload.get("response_style") or {}, default_value="")
        hard_constraints = ProfileField.from_dict(payload.get("hard_constraints") or {}, default_value=[])
        user_role_level = ProfileField.from_dict(payload.get("user_role_level") or {}, default_value="")
        project_context = ProfileField.from_dict(
            payload.get("project_context") or {},
            default_value=ProjectContext(project_name="", goals=[], key_decisions=[]),
        )
        project_context.value = ProjectContext.from_any(project_context.value)

        extra_fields_raw = payload.get("extra_fields") or {}
        extra_fields: dict[str, ProfileField] = {}
        if isinstance(extra_fields_raw, dict):
            for key, val in extra_fields_raw.items():
                field_obj = ProfileField.from_dict(val if isinstance(val, dict) else {}, default_value="")
                extra_fields[str(key)] = field_obj

        conflicts_raw = payload.get("conflicts") or []
        conflicts: list[ProfileConflict] = []
        if isinstance(conflicts_raw, list):
            for item in conflicts_raw:
                if isinstance(item, dict):
                    conflicts.append(ProfileConflict.from_dict(item))

        base.stack_tools = stack_tools
        base.response_style = response_style
        base.hard_constraints = hard_constraints
        base.user_role_level = user_role_level
        base.project_context = project_context
        base.extra_fields = extra_fields
        base.conflicts = conflicts
        return base


@dataclass
class ShortTermMessage:
    session_id: str
    role: str
    content: str
    timestamp: str


@dataclass
class TaskArtifact:
    step: str
    type: ArtifactType
    ref: str

    def to_dict(self) -> dict[str, str]:
        return {
            "step": str(self.step or ""),
            "type": self.type.value,
            "ref": str(self.ref or ""),
        }

    @classmethod
    def from_any(cls, value: Any) -> "TaskArtifact":
        if isinstance(value, TaskArtifact):
            return cls(step=value.step, type=value.type, ref=value.ref)
        if isinstance(value, dict):
            step = str(value.get("step") or "")
            ref = str(value.get("ref") or "")
            type_raw = str(value.get("type") or ArtifactType.LEGACY.value).strip().lower()
            try:
                art_type = ArtifactType(type_raw)
            except ValueError:
                art_type = ArtifactType.LEGACY
            return cls(step=step, type=art_type, ref=ref)
        return cls(step="", type=ArtifactType.LEGACY, ref=str(value or ""))


@dataclass
class TaskContext:
    session_id: str
    task_id: str
    task: str
    state: TaskState = TaskState.PLANNING
    plan: list[str] = field(default_factory=list)
    current_step: str | None = None
    done: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    artifacts: list[TaskArtifact] = field(default_factory=list)
    vars: dict[str, Any] = field(default_factory=dict)
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    # Backward-compat alias (legacy code still references .goal)
    @property
    def goal(self) -> str:
        return self.task

    @goal.setter
    def goal(self, value: str) -> None:
        self.task = str(value or "")

    # Backward-compat alias (legacy code still references .done_steps)
    @property
    def done_steps(self) -> list[str]:
        return self.done

    @done_steps.setter
    def done_steps(self, value: list[str]) -> None:
        self.done = [str(x) for x in (value or [])]

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "task_id": self.task_id,
            "task": self.task,
            "goal": self.task,  # compatibility key
            "state": self.state.value,
            "plan": list(self.plan),
            "current_step": self.current_step,
            "done": list(self.done),
            "done_steps": list(self.done),  # compatibility key
            "open_questions": list(self.open_questions),
            "artifacts": [a.to_dict() for a in self.artifacts],
            "vars": dict(self.vars),
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TaskContext":
        current_step_raw = payload.get("current_step")
        current_step = None if current_step_raw in (None, "") else str(current_step_raw)
        done = payload.get("done")
        if done is None:
            done = payload.get("done_steps") or []
        artifacts_raw = payload.get("artifacts") or []
        return cls(
            session_id=str(payload.get("session_id") or ""),
            task_id=str(payload.get("task_id") or ""),
            task=str(payload.get("task") or payload.get("goal") or ""),
            state=TaskState(str(payload.get("state") or TaskState.PLANNING.value)),
            plan=[str(x) for x in (payload.get("plan") or [])],
            current_step=current_step,
            done=[str(x) for x in (done or [])],
            open_questions=[str(x) for x in (payload.get("open_questions") or [])],
            artifacts=[TaskArtifact.from_any(x) for x in artifacts_raw],
            vars=dict(payload.get("vars") or {}),
            updated_at=str(payload.get("updated_at") or datetime.utcnow().isoformat()),
        )


@dataclass
class MemoryWriteEvent:
    layer: str
    keys: list[str]
    reason: str = ""
