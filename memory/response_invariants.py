from __future__ import annotations

from dataclasses import dataclass, field
import logging
import re
from typing import Any

TASK_STATES = {"PLANNING", "EXECUTION", "VALIDATION", "DONE"}
logger = logging.getLogger("memory")

_STEP_DONE_RE = re.compile(r"\[STEP_DONE:\s*(\d+)\]", flags=re.IGNORECASE)
_NEXT_STATE_RE = re.compile(r"\[NEXT_STATE:\s*([A-Z_]+)\]", flags=re.IGNORECASE)
_VALIDATION_OK_RE = re.compile(r"\[VALIDATION_OK\]", flags=re.IGNORECASE)
_VALIDATION_FAIL_RE = re.compile(r"\[VALIDATION_FAIL:\s*([^\]]+)\]", flags=re.IGNORECASE)
_DONE_RE = re.compile(r"\[DONE:\s*([^\]]+)\]", flags=re.IGNORECASE)
_OPEN_QUESTION_RE = re.compile(r"\[OPEN_QUESTION:\s*([^\]]+)\]", flags=re.IGNORECASE)
_CODE_ARTIFACT_RE = re.compile(r"\[CODE_ARTIFACT(?:\s*:\s*([^\]]+))?\]", flags=re.IGNORECASE)
_INTERNAL_TAG_RE = re.compile(r"<internal>(.*?)</internal>", flags=re.IGNORECASE | re.DOTALL)
_EXTERNAL_TAG_RE = re.compile(r"<external>(.*?)</external>", flags=re.IGNORECASE | re.DOTALL)
_CODE_FENCE_RE = re.compile(r"```(?:[\w+-]*)\n[\s\S]*?```", flags=re.MULTILINE)


@dataclass(frozen=True)
class ResponseSignals:
    has_code_artifact: bool
    has_done_phrase: bool
    has_open_question: bool


@dataclass
class ParsedMarkers:
    internal: str
    external: str
    step_done: int | None
    next_state: str | None
    next_state_raw: str | None
    next_state_marker_present: bool
    validation_ok: bool
    validation_fail_reason: str
    done_note: str | None
    open_question_note: str | None
    code_artifact_note: str | None
    format: str

    def has_validation_marker(self) -> bool:
        return bool(self.validation_ok or self.validation_fail_reason)

    def to_response_signals(self) -> ResponseSignals:
        return ResponseSignals(
            has_code_artifact=bool(self.code_artifact_note),
            has_done_phrase=bool(self.done_note),
            has_open_question=bool(self.open_question_note),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "internal": self.internal,
            "external": self.external,
            "step_done": self.step_done,
            "next_state": self.next_state,
            "next_state_raw": self.next_state_raw,
            "next_state_marker_present": self.next_state_marker_present,
            "validation_ok": self.validation_ok,
            "validation_fail_reason": self.validation_fail_reason,
            "done_note": self.done_note,
            "open_question_note": self.open_question_note,
            "code_artifact_note": self.code_artifact_note,
            "format": self.format,
        }


@dataclass
class ValidationReport:
    overall_status: str
    violations: list[dict[str, str]] = field(default_factory=list)
    can_retry: bool = False
    explanation: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall_status": self.overall_status,
            "violations": list(self.violations),
            "can_retry": bool(self.can_retry),
            "explanation": self.explanation,
        }


def parse_response_markers(raw_text: str) -> ParsedMarkers:
    raw = str(raw_text or "")

    internal_match = _INTERNAL_TAG_RE.search(raw)
    internal = str(internal_match.group(1) or "").strip() if internal_match else ""

    external_match = _EXTERNAL_TAG_RE.search(raw)
    if external_match:
        external = str(external_match.group(1) or "").strip()
    else:
        external = _INTERNAL_TAG_RE.sub("", raw)
        external = re.sub(r"</?external>", "", external, flags=re.IGNORECASE)
        external = external.strip()

    step_match = _STEP_DONE_RE.search(raw)
    next_state_match = _NEXT_STATE_RE.search(raw)
    validation_fail_match = _VALIDATION_FAIL_RE.search(raw)
    done_match = _DONE_RE.search(raw)
    open_question_match = _OPEN_QUESTION_RE.search(raw)
    code_artifact_match = _CODE_ARTIFACT_RE.search(raw)
    validation_ok = bool(_VALIDATION_OK_RE.search(raw))

    step_done: int | None = None
    if step_match:
        try:
            step_done = int(step_match.group(1))
        except Exception:
            step_done = None

    next_state_marker_present = bool(next_state_match)
    next_state_raw = str(next_state_match.group(1) or "").strip().upper() if next_state_match else None
    next_state: str | None = None
    if next_state_raw:
        if next_state_raw in TASK_STATES:
            next_state = next_state_raw
        else:
            logger.warning(
                "[MARKER_PARSE] Неизвестный NEXT_STATE: '%s' — маркер проигнорирован",
                next_state_raw,
            )
    fail_reason = str(validation_fail_match.group(1) or "").strip() if validation_fail_match else ""
    done_note = str(done_match.group(1) or "").strip() if done_match else None
    if done_note:
        logger.info("[MARKER_PARSE] done_note=%s", done_note)
    open_question_note = str(open_question_match.group(1) or "").strip() if open_question_match else None
    if open_question_note:
        logger.info("[MARKER_PARSE] open_question=%s", open_question_note)
    code_artifact_note: str | None = None
    if code_artifact_match:
        note = str(code_artifact_match.group(1) or "").strip()
        code_artifact_note = note or "present"
    if code_artifact_note is None and _CODE_FENCE_RE.search(external):
        code_artifact_note = "fenced_code"

    has_markers = bool(
        step_match
        or next_state_match
        or validation_ok
        or validation_fail_match
        or done_match
        or open_question_match
        or code_artifact_match
    )
    format_name = "markers" if has_markers else "legacy"

    cleaned_external = _strip_markers(external)
    if not cleaned_external and not external_match:
        cleaned_external = _strip_markers(raw)

    return ParsedMarkers(
        internal=internal,
        external=cleaned_external.strip(),
        step_done=step_done,
        next_state=next_state,
        next_state_raw=next_state_raw,
        next_state_marker_present=next_state_marker_present,
        validation_ok=validation_ok,
        validation_fail_reason=fail_reason,
        done_note=done_note,
        open_question_note=open_question_note,
        code_artifact_note=code_artifact_note,
        format=format_name,
    )


def validate_markers_present(parsed: ParsedMarkers) -> list[dict[str, str]]:
    violations: list[dict[str, str]] = []
    if parsed.step_done is None:
        violations.append({"code": "MISSING_STEP_DONE", "message": "Отсутствует маркер [STEP_DONE: N]."})
    if not parsed.next_state_marker_present:
        violations.append({"code": "MISSING_NEXT_STATE", "message": "Отсутствует маркер [NEXT_STATE: X]."})
    if not parsed.has_validation_marker():
        violations.append(
            {
                "code": "MISSING_VALIDATION_MARKER",
                "message": "Отсутствует [VALIDATION_OK] или [VALIDATION_FAIL: причина].",
            }
        )
    return violations


def validate_next_state_transition_allowed(
    *,
    parsed: ParsedMarkers,
    state_object: dict[str, Any],
    allowed_transitions: dict[str, set[str]],
) -> list[dict[str, str]]:
    current_state = str(state_object.get("state") or "").strip().upper()
    next_state = str(parsed.next_state or "").strip().upper()
    if not current_state or not next_state:
        return []
    if current_state not in TASK_STATES:
        return [{"code": "UNKNOWN_CURRENT_STATE", "message": f"Неизвестное текущее состояние: {current_state}."}]
    if next_state not in TASK_STATES:
        return [{"code": "UNKNOWN_NEXT_STATE", "message": f"Неизвестное следующее состояние: {next_state}."}]
    if next_state == current_state:
        return []
    if next_state not in set(allowed_transitions.get(current_state, set())):
        return [
            {
                "code": "FORBIDDEN_TRANSITION",
                "message": f"Недопустимый переход состояния: {current_state} -> {next_state}.",
            }
        ]
    return []


def validate_state_object_required_fields(state_object: dict[str, Any]) -> list[dict[str, str]]:
    required = ("task", "state", "plan", "current_step", "done", "artifacts", "open_questions")
    missing = [name for name in required if name not in state_object]
    if not missing:
        return []
    return [
        {
            "code": "STATE_OBJECT_MISSING_FIELDS",
            "message": f"В STATE OBJECT отсутствуют обязательные поля: {', '.join(missing)}.",
        }
    ]


def validate_step_scope_and_progress(*, parsed: ParsedMarkers, state_object: dict[str, Any]) -> list[dict[str, str]]:
    violations: list[dict[str, str]] = []
    state = str(state_object.get("state") or "").strip().upper()
    plan = state_object.get("plan") if isinstance(state_object.get("plan"), list) else []
    done = state_object.get("done") if isinstance(state_object.get("done"), list) else []
    current_step_raw = state_object.get("current_step")
    current_step = None if current_step_raw in (None, "") else str(current_step_raw)

    if state == "EXECUTION" and plan and done != plan and not current_step:
        violations.append(
            {
                "code": "EXECUTION_WITHOUT_CURRENT_STEP",
                "message": "В EXECUTION должен быть задан current_step до завершения всех шагов.",
            }
        )

    if current_step and current_step not in [str(x) for x in plan]:
        violations.append(
            {
                "code": "CURRENT_STEP_NOT_IN_PLAN",
                "message": "current_step отсутствует в plan.",
            }
        )

    if parsed.step_done is not None and parsed.step_done != len(done):
        violations.append(
            {
                "code": "STEP_DONE_MISMATCH",
                "message": f"Маркер STEP_DONE={parsed.step_done} не совпадает с done={len(done)}.",
            }
        )
    return violations


def validate_planning_without_implementation(*, parsed: ParsedMarkers, state_object: dict[str, Any]) -> list[dict[str, str]]:
    state = str(state_object.get("state") or "").strip().upper()
    if state != "PLANNING":
        return []
    external = str(parsed.external or "")
    if not external:
        return []

    looks_like_code = bool(
        "```" in external
        or re.search(r"\bextension\s+[A-Za-z_][A-Za-z0-9_]*\s*\{", external)
        or re.search(r"\bfunc\s+[A-Za-z_][A-Za-z0-9_]*\s*\(", external)
        or re.search(r"\bsubscript\s*\(", external)
        or re.search(r"\bimport\s+(SwiftUI|Foundation|UIKit)\b", external, flags=re.IGNORECASE)
    )
    if not looks_like_code:
        return []
    return [
        {
            "code": "PLANNING_CONTAINS_IMPLEMENTATION",
            "message": "В состоянии PLANNING нельзя выдавать реализацию кода до перехода в EXECUTION.",
        }
    ]


def validate_response(
    *,
    parsed: ParsedMarkers,
    state_object: dict[str, Any],
    allowed_transitions: dict[str, set[str]],
) -> ValidationReport:
    violations: list[dict[str, str]] = []
    violations.extend(validate_markers_present(parsed))
    violations.extend(validate_state_object_required_fields(state_object))
    violations.extend(
        validate_next_state_transition_allowed(
            parsed=parsed,
            state_object=state_object,
            allowed_transitions=allowed_transitions,
        )
    )
    violations.extend(validate_step_scope_and_progress(parsed=parsed, state_object=state_object))
    violations.extend(validate_planning_without_implementation(parsed=parsed, state_object=state_object))

    if _STEP_DONE_RE.search(parsed.external) or _NEXT_STATE_RE.search(parsed.external):
        violations.append(
            {
                "code": "MARKER_LEAK",
                "message": "Служебные маркеры утекли в user-facing текст.",
            }
        )

    if not violations:
        return ValidationReport(
            overall_status="ok",
            violations=[],
            can_retry=False,
            explanation="Ответ прошёл пост-валидацию инвариантов.",
        )

    details = "; ".join(str(item.get("message") or "").strip() for item in violations if str(item.get("message") or "").strip())
    explanation = "Не удалось подготовить ответ: нарушены инварианты."
    if details:
        explanation += f" {details}"
    return ValidationReport(
        overall_status="fail",
        violations=violations,
        can_retry=True,
        explanation=explanation,
    )


def _strip_markers(text: str) -> str:
    out = str(text or "")
    out = _STEP_DONE_RE.sub("", out)
    out = _NEXT_STATE_RE.sub("", out)
    out = _VALIDATION_OK_RE.sub("", out)
    out = _VALIDATION_FAIL_RE.sub("", out)
    out = _DONE_RE.sub("", out)
    out = _OPEN_QUESTION_RE.sub("", out)
    out = _CODE_ARTIFACT_RE.sub("", out)
    return "\n".join(line.rstrip() for line in out.splitlines()).strip()
