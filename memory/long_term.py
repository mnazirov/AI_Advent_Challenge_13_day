from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from typing import Any

import storage
from memory.models import LongTermProfile, ProfileConflict, ProfileField, ProfileSource, ProjectContext

logger = logging.getLogger("memory")


PROFILE_SIZE_LIMIT_BYTES = 8192
STACK_TOOLS_LIMIT = 12
HARD_CONSTRAINTS_LIMIT = 20
PROJECT_GOALS_LIMIT = 5
PROJECT_DECISIONS_LIMIT = 20
MAX_STRING_CHARS = 300
INFERRED_CONFIDENCE_THRESHOLD = 0.80


class LongTermMemory:
    CANONICAL_FIELDS = {
        "stack_tools",
        "response_style",
        "hard_constraints",
        "user_role_level",
        "project_context",
    }

    def get_profile(self, *, user_id: str) -> dict[str, Any]:
        profile = self._load_profile(user_id=user_id)
        return profile.to_dict()

    def update_profile_field(
        self,
        user_id: str,
        field: str,
        value: object,
        source: ProfileSource | str,
        confidence: float | None = None,
        verified: bool | None = None,
    ) -> str:
        source_enum = self._normalize_source(source)
        normalized_field = str(field or "").strip()
        if normalized_field not in self.CANONICAL_FIELDS:
            raise ValueError(f"Unknown profile field: {normalized_field}")

        if source_enum == ProfileSource.AGENT_INFERRED:
            effective_conf = float(confidence or 0.0)
            if effective_conf < INFERRED_CONFIDENCE_THRESHOLD:
                logger.info(
                    "[PROFILE_SKIP_LOW_CONFIDENCE] field=%s confidence=%.2f",
                    normalized_field,
                    effective_conf,
                )
                return "skipped_low_confidence"
            if verified is None:
                verified = False
        elif verified is None:
            verified = True

        profile = self._load_profile(user_id=user_id)
        current_field = self._get_field(profile, normalized_field)
        normalized_value = self._normalize_field_value(normalized_field, value)

        if (
            source_enum == ProfileSource.AGENT_INFERRED
            and bool(current_field.verified)
            and self._has_meaningful_value(current_field.value)
            and normalized_value != self._normalize_field_value(normalized_field, current_field.value)
        ):
            conflict = ProfileConflict(
                field=normalized_field,
                existing_value=current_field.value,
                inferred_value=normalized_value,
                confidence=float(confidence or 0.0),
                created_at=datetime.utcnow().isoformat(),
            )
            profile.conflicts.append(conflict)
            logger.info(
                "[PROFILE_CONFLICT] field=%s existing=%s inferred=%s",
                normalized_field,
                current_field.value,
                normalized_value,
            )
            self._enforce_profile_limits(profile, field=normalized_field)
            self._save_profile(user_id=user_id, profile=profile)
            return "conflict_recorded"

        new_field = ProfileField(
            value=normalized_value,
            source=source_enum,
            verified=bool(verified),
            confidence=float(confidence) if confidence is not None else None,
            updated_at=datetime.utcnow().isoformat(),
        )
        self._set_field(profile, normalized_field, new_field)
        self._enforce_profile_limits(profile, field=normalized_field)
        self._save_profile(user_id=user_id, profile=profile)
        logger.info(
            "[PROFILE_WRITE] field=%s source=%s verified=%s confidence=%s",
            normalized_field,
            source_enum.value,
            bool(verified),
            new_field.confidence,
        )
        return "updated"

    def delete_profile_field(self, user_id: str, field: str) -> None:
        normalized_field = str(field or "").strip()
        profile = self._load_profile(user_id=user_id)
        defaults = LongTermProfile.default()

        if normalized_field in self.CANONICAL_FIELDS:
            self._set_field(profile, normalized_field, self._get_field(defaults, normalized_field))
        else:
            profile.extra_fields.pop(normalized_field, None)

        self._enforce_profile_limits(profile, field=normalized_field or "extra_fields")
        self._save_profile(user_id=user_id, profile=profile)

    def add_profile_extra_field(
        self,
        user_id: str,
        field: str,
        value: object,
        source: ProfileSource | str,
    ) -> None:
        key = str(field or "").strip()
        if not key:
            raise ValueError("extra field name is empty")
        if key in self.CANONICAL_FIELDS:
            raise ValueError(f"Field '{key}' is canonical and cannot be created in extra_fields")

        source_enum = self._normalize_source(source)
        verified_default = source_enum in {ProfileSource.USER_EXPLICIT, ProfileSource.DEBUG_MENU}
        normalized_value = self._normalize_generic_value(value)
        profile = self._load_profile(user_id=user_id)
        profile.extra_fields[key] = ProfileField(
            value=normalized_value,
            source=source_enum,
            verified=verified_default,
            confidence=None,
            updated_at=datetime.utcnow().isoformat(),
        )
        self._enforce_profile_limits(profile, field=key)
        self._save_profile(user_id=user_id, profile=profile)
        logger.info(
            "[PROFILE_WRITE] field=%s source=%s verified=%s confidence=%s",
            key,
            source_enum.value,
            verified_default,
            None,
        )

    def confirm_profile_field(self, user_id: str, field: str) -> None:
        normalized_field = str(field or "").strip()
        profile = self._load_profile(user_id=user_id)
        profile_field = self._resolve_profile_field(profile, normalized_field)
        profile_field.verified = True
        profile_field.updated_at = datetime.utcnow().isoformat()
        self._save_profile(user_id=user_id, profile=profile)

    def resolve_profile_conflict(
        self,
        user_id: str,
        field: str,
        chosen_value: object | None = None,
        keep_existing: bool = False,
    ) -> None:
        if chosen_value is None and not keep_existing:
            raise ValueError("Either chosen_value or keep_existing=True must be provided")

        normalized_field = str(field or "").strip()
        profile = self._load_profile(user_id=user_id)
        conflict_index = self._find_conflict_index(profile.conflicts, normalized_field)
        if conflict_index < 0:
            raise ValueError(f"No conflict found for field '{normalized_field}'")

        if chosen_value is not None:
            normalized_value = self._normalize_field_value(normalized_field, chosen_value)
            self._set_field(
                profile,
                normalized_field,
                ProfileField(
                    value=normalized_value,
                    source=ProfileSource.USER_EXPLICIT,
                    verified=True,
                    confidence=None,
                    updated_at=datetime.utcnow().isoformat(),
                ),
            )

        profile.conflicts.pop(conflict_index)
        self._enforce_profile_limits(profile, field=normalized_field)
        self._save_profile(user_id=user_id, profile=profile)

    def add_decision(
        self,
        user_id: str,
        text: str,
        tags: list[str] | None = None,
        source: str = "user",
        ttl_days: int | None = None,
    ) -> dict:
        if source == "assistant":
            pending_id = storage.memory_add_longterm_pending(
                user_id=user_id,
                entry_type="decision",
                text=text,
                tags=tags or [],
                source=source,
                ttl_days=ttl_days,
            )
            return {"status": "pending", "pending_id": pending_id}
        if source not in {"user", "assistant_confirmed"}:
            raise ValueError("Unsupported source for long-term decision")
        storage.memory_add_longterm_decision(
            user_id=user_id,
            text=text,
            tags=tags or [],
            source=source,
            ttl_days=ttl_days,
            entry_type="decision",
        )
        return {"status": "committed"}

    def add_note(
        self,
        user_id: str,
        text: str,
        tags: list[str] | None = None,
        source: str = "user",
        ttl_days: int | None = 90,
    ) -> dict:
        if source == "assistant":
            pending_id = storage.memory_add_longterm_pending(
                user_id=user_id,
                entry_type="note",
                text=text,
                tags=tags or [],
                source=source,
                ttl_days=ttl_days,
            )
            return {"status": "pending", "pending_id": pending_id}
        if source not in {"user", "assistant_confirmed"}:
            raise ValueError("Unsupported source for long-term note")
        storage.memory_add_longterm_note(
            user_id=user_id,
            text=text,
            tags=tags or [],
            source=source,
            ttl_days=ttl_days,
            entry_type="note",
        )
        return {"status": "committed"}

    def retrieve(self, user_id: str, query: str, top_k: int = 3) -> dict:
        profile = self.get_profile(user_id=user_id)
        decisions = storage.memory_list_longterm_decisions(user_id=user_id, limit=100)
        notes = storage.memory_list_longterm_notes(user_id=user_id, limit=100)

        q_tokens = self._tokenize(query)
        top = max(1, int(top_k))
        ranked_decisions = sorted(decisions, key=lambda d: self._score_entry(d, q_tokens), reverse=True)[:top]
        ranked_notes = sorted(notes, key=lambda n: self._score_entry(n, q_tokens), reverse=True)[:top]

        decision_ids = [int(d["id"]) for d in ranked_decisions]
        note_ids = [int(n["id"]) for n in ranked_notes]

        return {
            "profile": profile,
            "decisions": ranked_decisions,
            "notes": ranked_notes,
            "read_meta": {
                "top_k": top,
                "decision_hits": len(ranked_decisions),
                "note_hits": len(ranked_notes),
                "decision_ids": decision_ids,
                "note_ids": note_ids,
                "decision_reason": "match/score",
                "note_reason": "match/score",
            },
        }

    def delete_decision(self, *, user_id: str, decision_id: int) -> bool:
        return storage.memory_delete_longterm_decision(user_id=user_id, decision_id=int(decision_id)) > 0

    def delete_note(self, *, user_id: str, note_id: int) -> bool:
        return storage.memory_delete_longterm_note(user_id=user_id, note_id=int(note_id)) > 0

    def propose_assistant_entry(
        self,
        *,
        user_id: str,
        entry_type: str,
        text: str,
        tags: list[str] | None = None,
        ttl_days: int | None = None,
    ) -> int:
        return storage.memory_add_longterm_pending(
            user_id=user_id,
            entry_type=entry_type,
            text=text,
            tags=tags or [],
            source="assistant",
            ttl_days=ttl_days,
        )

    def approve_pending_entry(self, *, user_id: str, pending_id: int) -> dict | None:
        pending = storage.memory_get_pending_by_id(user_id=user_id, pending_id=pending_id)
        if not pending or pending.get("status") != "pending":
            return None
        entry_type = str(pending.get("type") or "note")
        text = str(pending.get("text") or "")
        tags = list(pending.get("tags") or [])
        ttl_days = pending.get("ttl_days")

        if entry_type == "decision":
            self.add_decision(
                user_id=user_id,
                text=text,
                tags=tags,
                source="assistant_confirmed",
                ttl_days=ttl_days,
            )
        elif entry_type == "profile":
            profile = self._load_profile(user_id=user_id)
            ctx = ProjectContext.from_any(profile.project_context.value)
            if text and text not in ctx.key_decisions:
                ctx.key_decisions.append(text)
            self.update_profile_field(
                user_id=user_id,
                field="project_context",
                value=ctx,
                source=ProfileSource.USER_EXPLICIT,
                verified=True,
            )
        else:
            self.add_note(
                user_id=user_id,
                text=text,
                tags=tags,
                source="assistant_confirmed",
                ttl_days=ttl_days if ttl_days is not None else 90,
            )

        storage.memory_mark_pending_approved(user_id=user_id, pending_id=pending_id)
        return {"status": "approved", "pending_id": pending_id, "type": entry_type}

    def _load_profile(self, *, user_id: str) -> LongTermProfile:
        raw = storage.memory_load_longterm_profile(user_id=user_id) or {}
        if not raw:
            return LongTermProfile.default()
        return LongTermProfile.from_dict(raw)

    def _save_profile(self, *, user_id: str, profile: LongTermProfile) -> None:
        storage.memory_upsert_longterm_profile(
            user_id=user_id,
            profile=profile,
            conflicts=[c.to_dict() for c in profile.conflicts],
            source=ProfileSource.USER_EXPLICIT.value,
        )

    def _normalize_source(self, source: ProfileSource | str) -> ProfileSource:
        if isinstance(source, ProfileSource):
            return source
        source_raw = str(source or "").strip()
        try:
            return ProfileSource(source_raw)
        except ValueError as exc:
            raise ValueError(f"Unknown source: {source_raw}") from exc

    def _normalize_field_value(self, field: str, value: object) -> object:
        if field == "stack_tools":
            items = self._normalize_list(value)
            if len(items) > STACK_TOOLS_LIMIT:
                raise ValueError("stack_tools exceeds limit of 12")
            return items
        if field == "hard_constraints":
            items = self._normalize_list(value)
            if len(items) > HARD_CONSTRAINTS_LIMIT:
                raise ValueError("hard_constraints exceeds limit of 20")
            return items
        if field == "response_style":
            return self._truncate_str(value)
        if field == "user_role_level":
            return self._truncate_str(value)
        if field == "project_context":
            project_context = ProjectContext.from_any(value)
            project_context.project_name = self._truncate_str(project_context.project_name)
            project_context.goals = self._normalize_list(project_context.goals)
            project_context.key_decisions = self._normalize_list(project_context.key_decisions)
            if len(project_context.goals) > PROJECT_GOALS_LIMIT:
                raise ValueError("project_context.goals exceeds limit of 5")
            if len(project_context.key_decisions) > PROJECT_DECISIONS_LIMIT:
                raise ValueError("project_context.key_decisions exceeds limit of 20")
            return project_context
        raise ValueError(f"Unknown profile field: {field}")

    def _normalize_generic_value(self, value: object) -> object:
        if isinstance(value, str):
            return self._truncate_str(value)
        if isinstance(value, list):
            return [self._normalize_generic_value(x) for x in value]
        if isinstance(value, dict):
            return {str(k): self._normalize_generic_value(v) for k, v in value.items()}
        return value

    def _truncate_str(self, value: object) -> str:
        return str(value or "")[:MAX_STRING_CHARS]

    def _normalize_list(self, value: object) -> list[str]:
        if not isinstance(value, list):
            return []
        out: list[str] = []
        for item in value:
            normalized = self._truncate_str(item).strip()
            if normalized and normalized not in out:
                out.append(normalized)
        return out

    def _enforce_profile_limits(self, profile: LongTermProfile, *, field: str) -> None:
        stack_tools = self._normalize_list(profile.stack_tools.value)
        hard_constraints = self._normalize_list(profile.hard_constraints.value)
        project_context = ProjectContext.from_any(profile.project_context.value)
        project_context.project_name = self._truncate_str(project_context.project_name)
        project_context.goals = self._normalize_list(project_context.goals)
        project_context.key_decisions = self._normalize_list(project_context.key_decisions)

        if len(stack_tools) > STACK_TOOLS_LIMIT:
            raise ValueError("stack_tools exceeds limit of 12")
        if len(hard_constraints) > HARD_CONSTRAINTS_LIMIT:
            raise ValueError("hard_constraints exceeds limit of 20")
        if len(project_context.goals) > PROJECT_GOALS_LIMIT:
            raise ValueError("project_context.goals exceeds limit of 5")
        if len(project_context.key_decisions) > PROJECT_DECISIONS_LIMIT:
            raise ValueError("project_context.key_decisions exceeds limit of 20")

        profile.stack_tools.value = stack_tools
        profile.hard_constraints.value = hard_constraints
        profile.project_context.value = project_context
        profile.response_style.value = self._truncate_str(profile.response_style.value)
        profile.user_role_level.value = self._truncate_str(profile.user_role_level.value)

        serialized = json.dumps(profile.to_dict(), ensure_ascii=False).encode("utf-8")
        if len(serialized) > PROFILE_SIZE_LIMIT_BYTES:
            raise ValueError(
                f"Profile size limit exceeded: {len(serialized)}/{PROFILE_SIZE_LIMIT_BYTES} bytes.\n"
                f"                Cannot add field '{field}'. Remove or trim existing fields."
            )

    def _resolve_profile_field(self, profile: LongTermProfile, field: str) -> ProfileField:
        if field in self.CANONICAL_FIELDS:
            return self._get_field(profile, field)
        if field in profile.extra_fields:
            return profile.extra_fields[field]
        raise ValueError(f"Unknown profile field: {field}")

    def _get_field(self, profile: LongTermProfile, field: str) -> ProfileField:
        return getattr(profile, field)

    def _set_field(self, profile: LongTermProfile, field: str, value: ProfileField) -> None:
        if field in self.CANONICAL_FIELDS:
            setattr(profile, field, value)
            return
        profile.extra_fields[field] = value

    def _find_conflict_index(self, conflicts: list[ProfileConflict], field: str) -> int:
        for idx, conflict in enumerate(conflicts):
            if conflict.field == field:
                return idx
        return -1

    def _has_meaningful_value(self, value: object) -> bool:
        if value is None:
            return False
        if isinstance(value, str):
            return bool(value.strip())
        if isinstance(value, list):
            return bool(value)
        if isinstance(value, dict):
            return any(self._has_meaningful_value(v) for v in value.values())
        if isinstance(value, ProjectContext):
            return bool(value.project_name.strip() or value.goals or value.key_decisions)
        return True

    def _score_entry(self, entry: dict, q_tokens: set[str]) -> tuple[int, int]:
        text = str(entry.get("text") or "")
        tags = entry.get("tags") or []
        tokens = self._tokenize(text + " " + " ".join(str(t) for t in tags))
        overlap = len(q_tokens.intersection(tokens))
        tag_overlap = len(q_tokens.intersection({str(t).lower() for t in tags}))
        like_bonus = 0
        lowered = text.lower()
        if q_tokens:
            like_bonus = sum(1 for token in q_tokens if token in lowered)
        created_at = str(entry.get("created_at") or "")
        recency = int("".join(ch for ch in created_at if ch.isdigit())[:8] or 0)
        return overlap + tag_overlap + like_bonus, recency

    def _tokenize(self, text: str) -> set[str]:
        raw = re.split(r"[^\wа-яА-ЯёЁ]+", (text or "").lower())
        return {t for t in raw if len(t) > 2}
