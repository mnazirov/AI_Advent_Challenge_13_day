from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class IntentName(str, Enum):
    TASK_INTENT = "task_intent"
    PLAN_FORMATION_INTENT = "plan_formation_intent"
    DECISION_MEMORY_WRITE = "decision_memory_write"
    NOTE_MEMORY_WRITE = "note_memory_write"

    START_EXECUTION = "start_execution"
    PLAN_APPROVED = "plan_approved"
    PLAN_FORMATION = "plan_formation"
    SKIP_MANDATORY_PLANNING = "skip_mandatory_planning"
    GOAL_CLARIFICATION = "goal_clarification"
    DIRECT_CODE_REQUEST = "direct_code_request"

    VALIDATION_REQUEST = "validation_request"
    VALIDATION_CHECKLIST_REQUEST = "validation_checklist_request"
    VALIDATION_CONFIRM = "validation_confirm"
    VALIDATION_REJECT = "validation_reject"
    VALIDATION_SKIP_REQUEST = "validation_skip_request"

    YES_CONFIRMATION = "yes_confirmation"
    NO_CONFIRMATION = "no_confirmation"
    STACK_SWITCH_REQUEST = "stack_switch_request"
    THIRD_PARTY_DEPENDENCY_REQUEST = "third_party_dependency_request"

    STEP_COMPLETED = "step_completed"
    WORKING_UPDATE = "working_update"
    CONFIRM_PENDING_MEMORY = "confirm_pending_memory"


class IntentOutcome(str, Enum):
    MATCH = "match"
    NO_MATCH = "no_match"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class IntentDecision:
    intent: IntentName
    outcome: IntentOutcome
    confidence: float = 0.0
    reason: str = ""
    reason_code: str = ""

    @property
    def is_match(self) -> bool:
        return self.outcome == IntentOutcome.MATCH

    @property
    def is_no_match(self) -> bool:
        return self.outcome == IntentOutcome.NO_MATCH

    @property
    def is_unknown(self) -> bool:
        return self.outcome == IntentOutcome.UNKNOWN

    @property
    def status(self) -> str:
        return self.outcome.value

    @staticmethod
    def _clamp_confidence(value: float) -> float:
        try:
            num = float(value)
        except Exception:
            num = 0.0
        return max(0.0, min(1.0, num))

    @classmethod
    def match(
        cls,
        *,
        intent: IntentName,
        confidence: float,
        reason: str = "",
        reason_code: str = "",
    ) -> "IntentDecision":
        return cls(
            intent=intent,
            outcome=IntentOutcome.MATCH,
            confidence=cls._clamp_confidence(confidence),
            reason=str(reason or ""),
            reason_code=str(reason_code or ""),
        )

    @classmethod
    def no_match(
        cls,
        *,
        intent: IntentName,
        confidence: float,
        reason: str = "",
        reason_code: str = "",
    ) -> "IntentDecision":
        return cls(
            intent=intent,
            outcome=IntentOutcome.NO_MATCH,
            confidence=cls._clamp_confidence(confidence),
            reason=str(reason or ""),
            reason_code=str(reason_code or ""),
        )

    @classmethod
    def unknown(
        cls,
        *,
        intent: IntentName,
        reason: str = "",
        reason_code: str = "",
        confidence: float = 0.0,
    ) -> "IntentDecision":
        return cls(
            intent=intent,
            outcome=IntentOutcome.UNKNOWN,
            confidence=cls._clamp_confidence(confidence),
            reason=str(reason or ""),
            reason_code=str(reason_code or ""),
        )


def parse_client_intent(raw: Any) -> tuple[IntentName | None, dict[str, Any]]:
    if not isinstance(raw, dict):
        return None, {}
    raw_name = raw.get("intent")
    if raw_name in (None, ""):
        raw_name = raw.get("type")
    if raw_name in (None, ""):
        raw_name = raw.get("name")
    name = str(raw_name or "").strip().lower()
    payload = raw.get("payload")
    if not isinstance(payload, dict):
        payload = {}
    if not name:
        return None, payload
    try:
        return IntentName(name), payload
    except ValueError:
        return None, payload
