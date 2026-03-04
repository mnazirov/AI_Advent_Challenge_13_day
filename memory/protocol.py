from __future__ import annotations

from datetime import datetime
import json
import logging
import re
import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any

from memory.models import ProfileSource

if TYPE_CHECKING:
    from memory.long_term import LongTermMemory
    from memory.working import WorkingMemory

logger = logging.getLogger("memory.protocol")

PROTOCOL_VERSION_KEY = "protocol_profile_version"
PROTOCOL_PROFILE_KEY = "protocol_profile"
PROTOCOL_PENDING_KEY = "protocol_pending_update"
PROTOCOL_STATE_META_KEY = "protocol_state_meta"

IOS_PHASES = ["ONBOARDING", "PLANNING", "EXECUTION", "MONETIZATION", "LAUNCH", "GROWTH"]
PHASE_INDEX = {phase: idx for idx, phase in enumerate(IOS_PHASES)}

PHASE_NEXT_HINTS = {
    "ONBOARDING": "коротко уточнить идею приложения и для кого вы его делаете",
    "PLANNING": "зафиксировать MVP, ключевые экраны и архитектуру",
    "EXECUTION": "собрать первый рабочий флоу в SwiftUI и проверить его на устройстве",
    "MONETIZATION": "подключить подписки (StoreKit 2 или RevenueCat) и протестировать paywall",
    "LAUNCH": "подготовить релиз в App Store (метаданные, скриншоты, политика)",
    "GROWTH": "настроить аналитику и выбрать одну гипотезу роста на следующую итерацию",
}

PHASE_INTERNAL_GUIDANCE = {
    "ONBOARDING": "Collect experience level, app idea, audience, and constraints naturally from conversation.",
    "PLANNING": "Define MVP scope, screen map, architecture baseline, and acceptance milestones.",
    "EXECUTION": "Drive incremental implementation with practical code and small review loops.",
    "MONETIZATION": "Design conversion flow, paywall, subscription logic, and trial/offer strategy.",
    "LAUNCH": "Prepare App Store assets, QA checklist, privacy/compliance details, and release sequence.",
    "GROWTH": "Focus on analytics, funnel bottlenecks, conversion optimization, and iteration cadence.",
}


class ProtocolCoordinator:
    def __init__(self, *, long_term: "LongTermMemory", working: "WorkingMemory", repo_root: Path | None = None):
        self.long_term = long_term
        self.working = working
        self.repo_root = Path(repo_root or Path(__file__).resolve().parents[1])

    def ensure_protocol_profile(self, *, user_id: str) -> dict[str, Any]:
        profile = self.long_term.get_profile(user_id=user_id) or {}
        extra = self._extra_fields(profile)

        if PROTOCOL_VERSION_KEY not in extra:
            self._save_extra(user_id=user_id, key=PROTOCOL_VERSION_KEY, value="v2.0")
            logger.info("[PROTOCOL_PROFILE_INIT] key=%s value=v2.0", PROTOCOL_VERSION_KEY)
        if PROTOCOL_PROFILE_KEY not in extra:
            self._save_extra(user_id=user_id, key=PROTOCOL_PROFILE_KEY, value=self._default_protocol_profile())
            logger.info("[PROTOCOL_PROFILE_INIT] key=%s", PROTOCOL_PROFILE_KEY)
        if PROTOCOL_PENDING_KEY not in extra:
            self._save_extra(user_id=user_id, key=PROTOCOL_PENDING_KEY, value={})
            logger.info("[PROTOCOL_PROFILE_INIT] key=%s value=empty", PROTOCOL_PENDING_KEY)
        if PROTOCOL_STATE_META_KEY not in extra:
            self._save_extra(user_id=user_id, key=PROTOCOL_STATE_META_KEY, value=self._default_state_meta())
            logger.info("[PROTOCOL_PROFILE_INIT] key=%s", PROTOCOL_STATE_META_KEY)

        return self.get_protocol_status(user_id=user_id)

    def get_protocol_status(self, *, user_id: str) -> dict[str, Any]:
        profile = self.long_term.get_profile(user_id=user_id) or {}
        extra = self._extra_fields(profile)
        protocol_profile = self._safe_dict(extra.get(PROTOCOL_PROFILE_KEY)) or self._default_protocol_profile()
        pending = self._safe_dict(extra.get(PROTOCOL_PENDING_KEY)) or {}
        state_meta = self._safe_dict(extra.get(PROTOCOL_STATE_META_KEY)) or self._default_state_meta()
        version = str(extra.get(PROTOCOL_VERSION_KEY) or "v2.0").strip() or "v2.0"
        return {
            "protocol_profile_version": version,
            "protocol_profile": protocol_profile,
            "protocol_pending_update": pending,
            "protocol_state_meta": state_meta,
            "bootstrap_required": False,
        }

    def prepare_turn(self, *, session_id: str, user_id: str, user_message: str) -> dict[str, Any]:
        del session_id
        snapshot = self.ensure_protocol_profile(user_id=user_id)
        profile = dict(snapshot.get("protocol_profile") or {})
        state_meta = dict(snapshot.get("protocol_state_meta") or {})

        profile_patch = self._extract_profile_patch(user_message=user_message, current_profile=profile)
        if profile_patch:
            profile.update(profile_patch)

        phase_before = self._normalize_phase(state_meta.get("phase"))
        phase_after, transition_reason = self._advance_phase(
            current_phase=phase_before,
            user_message=user_message,
            profile=profile,
        )

        if phase_after != phase_before:
            state_meta["phase"] = phase_after
            history = [str(x) for x in (state_meta.get("phase_history") or []) if str(x)]
            if not history:
                history = [phase_before]
            if history[-1] != phase_after:
                history.append(phase_after)
            state_meta["phase_history"] = history
            state_meta["last_transition_reason"] = transition_reason
            state_meta["phase_updated_at"] = datetime.utcnow().isoformat()
            logger.info("[PROTOCOL_STATE_TRANSITION] %s -> %s reason=%s", phase_before, phase_after, transition_reason)

        invariant_report = self._evaluate_turn_invariants(
            phase=phase_after,
            user_message=user_message,
            profile=profile,
            state_meta=state_meta,
        )
        auto_fixes = self._apply_invariant_fixes(profile=profile, state_meta=state_meta, phase=phase_after, report=invariant_report)
        if auto_fixes:
            invariant_report["auto_fixes"] = auto_fixes
            invariant_report["overall_status"] = "corrected"

        state_meta["last_invariants"] = invariant_report
        state_meta["last_user_message_at"] = datetime.utcnow().isoformat()
        state_meta["last_user_message_preview"] = str(user_message or "").strip()[:180]

        self._save_extra(user_id=user_id, key=PROTOCOL_PROFILE_KEY, value=profile)
        self._save_extra(user_id=user_id, key=PROTOCOL_STATE_META_KEY, value=state_meta)

        next_step = self._next_step_hint(phase=phase_after)
        internal_context = self._build_internal_context(
            phase=phase_after,
            profile=profile,
            state_meta=state_meta,
            invariants=invariant_report,
            next_step=next_step,
        )
        return {
            "protocol_state": {
                "phase": phase_after,
                "phase_history": list(state_meta.get("phase_history") or []),
                "profile": profile,
                "state_meta": state_meta,
            },
            "invariant_report": invariant_report,
            "next_step": next_step,
            "internal_context": internal_context,
        }

    def propose_profile_update(
        self,
        *,
        user_id: str,
        updates: dict[str, Any],
        reason: str = "user_request",
    ) -> dict[str, Any]:
        snapshot = self.ensure_protocol_profile(user_id=user_id)
        current = dict(snapshot.get("protocol_profile") or {})
        candidate = dict(current)
        for key, value in dict(updates or {}).items():
            if key not in current:
                continue
            text = str(value or "").strip()
            if text:
                candidate[key] = text
        changed = [k for k in candidate.keys() if candidate.get(k) != current.get(k)]
        if not changed:
            return {
                "status": "no_changes",
                "from_version": snapshot["protocol_profile_version"],
                "to_version": snapshot["protocol_profile_version"],
                "changes": [],
                "pending_update": {},
            }

        from_version = str(snapshot["protocol_profile_version"])
        to_version = self._bump_minor(from_version)
        pending = {
            "from_version": from_version,
            "to_version": to_version,
            "changes": changed,
            "candidate": candidate,
            "reason": str(reason or "user_request"),
            "requested_at": datetime.utcnow().isoformat(),
        }
        self._save_extra(user_id=user_id, key=PROTOCOL_PENDING_KEY, value=pending)
        return {
            "status": "proposed",
            "from_version": from_version,
            "to_version": to_version,
            "changes": changed,
            "pending_update": pending,
        }

    def confirm_profile_update(self, *, user_id: str, accept: bool = True) -> dict[str, Any]:
        snapshot = self.ensure_protocol_profile(user_id=user_id)
        pending = self._safe_dict(snapshot.get("protocol_pending_update"))
        if not pending:
            return {"status": "no_pending_update"}

        if not accept:
            self._save_extra(user_id=user_id, key=PROTOCOL_PENDING_KEY, value={})
            return {"status": "rejected", "from_version": pending.get("from_version"), "to_version": pending.get("to_version")}

        candidate = self._safe_dict(pending.get("candidate")) or self._default_protocol_profile()
        self._save_extra(user_id=user_id, key=PROTOCOL_PROFILE_KEY, value=candidate)
        self._save_extra(user_id=user_id, key=PROTOCOL_VERSION_KEY, value=str(pending.get("to_version") or "v2.0"))
        self._save_extra(user_id=user_id, key=PROTOCOL_PENDING_KEY, value={})
        return {
            "status": "applied",
            "from_version": pending.get("from_version"),
            "to_version": pending.get("to_version"),
            "profile": candidate,
        }

    def set_protocol_state_meta(self, *, user_id: str, patch: dict[str, Any]) -> dict[str, Any]:
        snapshot = self.ensure_protocol_profile(user_id=user_id)
        state_meta = dict(snapshot.get("protocol_state_meta") or {})
        state_meta.update(dict(patch or {}))
        state_meta["updated_at"] = datetime.utcnow().isoformat()
        self._save_extra(user_id=user_id, key=PROTOCOL_STATE_META_KEY, value=state_meta)
        return state_meta

    def build_protocol_header(
        self,
        *,
        session_id: str,
        user_id: str,
        invariant_report: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        runtime = self.prepare_turn(session_id=session_id, user_id=user_id, user_message="")
        state = dict(runtime.get("protocol_state") or {})
        phase = self._normalize_phase(state.get("phase"))
        inv = invariant_report or runtime.get("invariant_report") or {}
        return {
            "header": f"phase={phase}",
            "next_step": runtime.get("next_step"),
            "protocol_state": state,
            "invariant_report": inv,
        }

    def evaluate_invariants(self, *, run_external: bool = False) -> dict[str, Any]:
        if not run_external:
            return {
                "overall_status": "internal_only",
                "results": [
                    {
                        "id": "I-CORE",
                        "status": "pass",
                        "message": "Инварианты этапа и согласованности проверяются внутри каждого хода.",
                    }
                ],
                "auto_fixes": [],
            }

        requirements_count = len(self._read_requirements())
        results: list[dict[str, Any]] = []
        if requirements_count <= 20:
            results.append({"id": "EXT-1", "status": "pass", "message": f"Найдено зависимостей: {requirements_count}."})
        else:
            results.append(
                {
                    "id": "EXT-1",
                    "status": "warn",
                    "message": f"Зависимостей: {requirements_count}. Стоит проверить необходимость всех пакетов.",
                }
            )

        secret_scan = self._run_secret_scan()
        results.append(secret_scan)
        overall = "pass"
        if any(r.get("status") == "blocked" for r in results):
            overall = "blocked"
        elif any(r.get("status") == "warn" for r in results):
            overall = "warning"
        elif any(r.get("status") == "requires_external_validation" for r in results):
            overall = "requires_external_validation"
        return {
            "overall_status": overall,
            "results": results,
            "auto_fixes": [],
        }

    def _extract_profile_patch(self, *, user_message: str, current_profile: dict[str, Any]) -> dict[str, Any]:
        text = str(user_message or "").strip()
        lower = text.lower()
        patch: dict[str, Any] = {}

        experience = self._detect_experience_level(lower)
        if experience:
            patch["experience_level"] = experience

        stack = self._detect_stack(lower)
        if stack:
            patch["stack"] = stack

        monetization = self._detect_monetization(lower)
        if monetization:
            patch["monetization_model"] = monetization

        idea = self._detect_app_idea(text)
        if idea:
            patch["app_idea"] = idea

        audience = self._detect_target_audience(text)
        if audience:
            patch["target_audience"] = audience

        if self._looks_like_progress_update(lower):
            patch["current_progress"] = text[:220]

        if not patch and not current_profile.get("stack"):
            patch["stack"] = "Swift + SwiftUI"
        return patch

    def _advance_phase(
        self,
        *,
        current_phase: str,
        user_message: str,
        profile: dict[str, Any],
    ) -> tuple[str, str]:
        text = str(user_message or "").lower()
        phase = self._normalize_phase(current_phase)

        explicit_phase = self._phase_from_explicit_request(text)
        if explicit_phase and PHASE_INDEX[explicit_phase] > PHASE_INDEX[phase]:
            return explicit_phase, "explicit_user_request"

        if phase == "ONBOARDING":
            if profile.get("app_idea") and (profile.get("target_audience") or profile.get("experience_level") != "unknown"):
                return "PLANNING", "profile_minimum_collected"
        elif phase == "PLANNING":
            if any(token in text for token in ["план готов", "mvp готов", "давай писать код", "начнем код", "xcode"]):
                return "EXECUTION", "implementation_started"
        elif phase == "EXECUTION":
            if any(token in text for token in ["подписк", "storekit", "revenuecat", "paywall", "монетизац"]):
                return "MONETIZATION", "monetization_focus_detected"
        elif phase == "MONETIZATION":
            if any(token in text for token in ["app store", "релиз", "запуск", "aso", "скриншот"]):
                return "LAUNCH", "launch_focus_detected"
        elif phase == "LAUNCH":
            if any(token in text for token in ["growth", "рост", "аналитик", "конверс", "итерац"]):
                return "GROWTH", "growth_focus_detected"
        return phase, "no_transition"

    def _evaluate_turn_invariants(
        self,
        *,
        phase: str,
        user_message: str,
        profile: dict[str, Any],
        state_meta: dict[str, Any],
    ) -> dict[str, Any]:
        del state_meta
        results: list[dict[str, str]] = []
        lower = str(user_message or "").lower()

        if not profile.get("stack"):
            results.append({"id": "WF-1", "status": "warn", "message": "Стек не зафиксирован; будет применён default Swift + SwiftUI."})
        else:
            results.append({"id": "WF-1", "status": "pass", "message": "Стек согласован."})

        if PHASE_INDEX.get(phase, 0) >= PHASE_INDEX["MONETIZATION"] and not profile.get("monetization_model"):
            results.append(
                {
                    "id": "WF-2",
                    "status": "warn",
                    "message": "Этап монетизации активен, но модель монетизации пока не закреплена.",
                }
            )
        else:
            results.append({"id": "WF-2", "status": "pass", "message": "Модель монетизации согласована с этапом."})

        if phase == "ONBOARDING" and any(token in lower for token in ["app store", "aso", "релиз"]):
            results.append(
                {
                    "id": "WF-3",
                    "status": "warn",
                    "message": "Запрос о релизе появился до планирования и реализации; ответ должен мягко вернуть к базовым шагам.",
                }
            )
        else:
            results.append({"id": "WF-3", "status": "pass", "message": "Порядок этапов не нарушен."})

        overall = "pass"
        if any(item.get("status") == "warn" for item in results):
            overall = "warning"
        return {"overall_status": overall, "results": results, "auto_fixes": []}

    def _apply_invariant_fixes(
        self,
        *,
        profile: dict[str, Any],
        state_meta: dict[str, Any],
        phase: str,
        report: dict[str, Any],
    ) -> list[str]:
        del state_meta
        fixes: list[str] = []
        if not profile.get("stack"):
            profile["stack"] = "Swift + SwiftUI"
            fixes.append("stack_default_applied")
        if PHASE_INDEX.get(phase, 0) >= PHASE_INDEX["MONETIZATION"] and not profile.get("monetization_model"):
            profile["monetization_model"] = "Подписка (черновой вариант)"
            fixes.append("monetization_default_applied")
        report["auto_fixes"] = fixes
        return fixes

    def _build_internal_context(
        self,
        *,
        phase: str,
        profile: dict[str, Any],
        state_meta: dict[str, Any],
        invariants: dict[str, Any],
        next_step: str,
    ) -> str:
        return (
            "[IOS_INTERNAL_WORKFLOW]\n"
            f"phase={phase}\n"
            f"phase_history={state_meta.get('phase_history')}\n"
            f"transition_reason={state_meta.get('last_transition_reason')}\n"
            f"profile={json.dumps(profile, ensure_ascii=False)}\n"
            f"invariants={json.dumps(invariants, ensure_ascii=False)}\n"
            f"phase_guidance={PHASE_INTERNAL_GUIDANCE.get(phase, '')}\n"
            f"suggested_next_step={next_step}\n"
            "Important: internal data is not user-facing. Put it only inside <internal>...</internal>."
        )

    def _next_step_hint(self, *, phase: str) -> str:
        return str(PHASE_NEXT_HINTS.get(phase) or PHASE_NEXT_HINTS["ONBOARDING"])

    @staticmethod
    def _detect_experience_level(lower_text: str) -> str:
        if any(token in lower_text for token in ["с нуля", "нович", "первое приложение", "нет опыта"]):
            return "новичок"
        if any(token in lower_text for token in ["senior", "продвинут", "lead", "10 лет", "архитект"]):
            return "продвинутый"
        if any(token in lower_text for token in ["есть опыт", "делал", "уже писал", "middle"]):
            return "есть опыт"
        return ""

    @staticmethod
    def _detect_stack(lower_text: str) -> str:
        if "swiftui" in lower_text:
            return "Swift + SwiftUI"
        if "uikit" in lower_text:
            return "Swift + UIKit"
        if "flutter" in lower_text:
            return "Flutter (cross-platform)"
        if "react native" in lower_text or "react-native" in lower_text:
            return "React Native (cross-platform)"
        return ""

    @staticmethod
    def _detect_monetization(lower_text: str) -> str:
        if any(token in lower_text for token in ["подписк", "subscription", "storekit", "revenuecat"]):
            return "подписка"
        if any(token in lower_text for token in ["реклам", "ads"]):
            return "реклама"
        if any(token in lower_text for token in ["one-time", "разовая покупка", "paid app"]):
            return "разовая покупка"
        if "freemium" in lower_text:
            return "freemium"
        return ""

    @staticmethod
    def _detect_app_idea(text: str) -> str:
        patterns = [
            r"(?:идея|хочу сделать|хочу создать|делаю)\s*(?:ios-?приложение|приложение)?\s*(?:для)?\s*[:\-]?\s*(.+)",
            r"(?:app idea)\s*[:\-]\s*(.+)",
        ]
        normalized = " ".join(str(text or "").split())
        for pattern in patterns:
            match = re.search(pattern, normalized, flags=re.IGNORECASE)
            if match:
                candidate = str(match.group(1) or "").strip(" .")
                if candidate:
                    return candidate[:220]
        return ""

    @staticmethod
    def _detect_target_audience(text: str) -> str:
        patterns = [
            r"(?:для|аудитория|целевая аудитория)\s*[:\-]?\s*(.+)",
            r"(?:target audience)\s*[:\-]\s*(.+)",
        ]
        normalized = " ".join(str(text or "").split())
        for pattern in patterns:
            match = re.search(pattern, normalized, flags=re.IGNORECASE)
            if not match:
                continue
            candidate = str(match.group(1) or "").strip(" .")
            if candidate and len(candidate.split()) <= 18:
                return candidate[:200]
        return ""

    @staticmethod
    def _looks_like_progress_update(lower_text: str) -> bool:
        tokens = [
            "сделал",
            "готово",
            "добавил",
            "дописал",
            "законч",
            "implemented",
            "done",
            "completed",
        ]
        return any(token in lower_text for token in tokens)

    def _phase_from_explicit_request(self, lower_text: str) -> str:
        mapping = {
            "onboarding": "ONBOARDING",
            "к онбордингу": "ONBOARDING",
            "planning": "PLANNING",
            "к планированию": "PLANNING",
            "execution": "EXECUTION",
            "к разработке": "EXECUTION",
            "к монетизации": "MONETIZATION",
            "monetization": "MONETIZATION",
            "к запуску": "LAUNCH",
            "launch": "LAUNCH",
            "к росту": "GROWTH",
            "growth": "GROWTH",
        }
        for token, phase in mapping.items():
            if token in lower_text:
                return phase
        return ""

    def _run_secret_scan(self) -> dict[str, Any]:
        if shutil.which("trufflehog"):
            command = ["trufflehog", "filesystem", ".", "--no-update"]
        elif shutil.which("git-secrets"):
            command = ["git-secrets", "--scan", "-r", "."]
        else:
            return {
                "id": "EXT-2",
                "status": "requires_external_validation",
                "message": "Сканер секретов не найден в окружении.",
            }
        try:
            proc = subprocess.run(
                command,
                cwd=str(self.repo_root),
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
        except Exception as exc:
            return {
                "id": "EXT-2",
                "status": "requires_external_validation",
                "message": f"Не удалось выполнить сканер секретов: {exc}",
            }
        if proc.returncode == 0:
            return {"id": "EXT-2", "status": "pass", "message": "Сканер секретов не выявил блокирующих проблем."}
        excerpt = ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()[:400]
        return {
            "id": "EXT-2",
            "status": "warn",
            "message": "Сканер секретов вернул предупреждения. Нужна ручная проверка.",
            "details": excerpt,
        }

    def _read_requirements(self) -> list[str]:
        req_file = self.repo_root / "requirements.txt"
        if not req_file.exists():
            return []
        lines = []
        for raw in req_file.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            lines.append(line)
        return lines

    def _save_extra(self, *, user_id: str, key: str, value: Any) -> None:
        self.long_term.add_profile_extra_field(
            user_id=user_id,
            field=key,
            value=value,
            source=ProfileSource.USER_EXPLICIT,
        )

    @staticmethod
    def _extra_fields(profile: dict[str, Any]) -> dict[str, Any]:
        extra = profile.get("extra_fields")
        if not isinstance(extra, dict):
            return {}
        out: dict[str, Any] = {}
        for key, payload in extra.items():
            if isinstance(payload, dict):
                out[str(key)] = payload.get("value")
        return out

    @staticmethod
    def _safe_dict(value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            return dict(value)
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                return {}
        return {}

    @staticmethod
    def _normalize_phase(value: object) -> str:
        raw = str(value or "").strip().upper()
        return raw if raw in PHASE_INDEX else "ONBOARDING"

    @staticmethod
    def _default_protocol_profile() -> dict[str, Any]:
        return {
            "experience_level": "unknown",
            "app_idea": "",
            "target_audience": "",
            "stack": "Swift + SwiftUI",
            "monetization_model": "",
            "current_progress": "",
            "updated_at": datetime.utcnow().isoformat(),
        }

    @staticmethod
    def _default_state_meta() -> dict[str, Any]:
        return {
            "phase": "ONBOARDING",
            "phase_history": ["ONBOARDING"],
            "last_transition_reason": "initial_state",
            "phase_updated_at": datetime.utcnow().isoformat(),
            "last_invariants": {},
        }

    @staticmethod
    def _bump_minor(version: str) -> str:
        raw = str(version or "v2.0").strip().lower()
        match = re.match(r"v?(\d+)\.(\d+)", raw)
        if not match:
            return "v2.1"
        major = int(match.group(1))
        minor = int(match.group(2)) + 1
        return f"v{major}.{minor}"
