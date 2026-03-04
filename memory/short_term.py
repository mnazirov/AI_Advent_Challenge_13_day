from __future__ import annotations

from datetime import datetime
import logging

import storage

logger = logging.getLogger("memory")

PREVIEW_LENGTH = 120
MAX_DEBUG_TEXT_CHARS = 4000
ALLOWED_ROLES = {"user", "assistant"}


class ShortTermMemory:
    def __init__(self, limit_n: int = 30):
        self.limit_n = max(1, int(limit_n))

    def append(self, session_id: str, role: str, content: str, timestamp: str | None = None) -> None:
        ts = timestamp or datetime.utcnow().isoformat()
        storage.memory_append_short_term_message(session_id=session_id, role=role, content=content, timestamp=ts)
        pruned = storage.memory_trim_short_term_messages(session_id=session_id, keep_last=self.limit_n)
        logger.info(
            "[MEMORY_WRITE] layer=short_term appended=1 pruned=%s session=%s",
            int(pruned),
            session_id,
        )

    def get_context(self, session_id: str) -> list[dict[str, str]]:
        rows = storage.memory_load_short_term_messages(session_id=session_id)
        context = [
            {"role": r["role"], "content": r["content"]}
            for r in rows[-self.limit_n :]
            if str(r.get("role") or "") in ALLOWED_ROLES
        ]
        logger.info("[MEMORY_READ] layer=short_term turns=%s", len(context))
        return context

    def clear_session(self, session_id: str) -> None:
        storage.memory_clear_short_term_messages(session_id=session_id)

    def hydrate(self, session_id: str, messages: list[dict[str, str]]) -> None:
        self.clear_session(session_id)
        for msg in messages[-self.limit_n :]:
            role = str(msg.get("role") or "")
            content = str(msg.get("content") or "")
            if role in ALLOWED_ROLES and content:
                self.append(session_id=session_id, role=role, content=content)

    def snapshot(self, session_id: str) -> dict:
        """Возвращает снимок short-term для debug: limit_n, turns_count, turns с preview/full/full_truncated."""
        rows = storage.memory_load_short_term_messages_for_debug(
            session_id=session_id, limit_n=self.limit_n
        )
        turns = []
        for r in rows:
            content = str(r.get("content") or "")
            preview = content[:PREVIEW_LENGTH] + ("…" if len(content) > PREVIEW_LENGTH else "")
            full = content[:MAX_DEBUG_TEXT_CHARS]
            full_truncated = len(content) > MAX_DEBUG_TEXT_CHARS
            if full_truncated:
                full = full + "…"
            turns.append(
                {
                    "id": r.get("id"),
                    "role": r.get("role", ""),
                    "timestamp": r.get("timestamp", ""),
                    "preview": preview,
                    "full": full,
                    "full_truncated": full_truncated,
                }
            )
        return {
            "limit_n": self.limit_n,
            "turns_count": len(turns),
            "turns": turns,
        }
