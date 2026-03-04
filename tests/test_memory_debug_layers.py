from __future__ import annotations

import unittest
from datetime import datetime
from uuid import uuid4

import app as webapp
import storage


class ShortTermSessionScopeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        storage.init_db()

    def _insert_short_term_row(self, *, session_id: str, runtime_id: str, role: str, content: str) -> None:
        with storage._get_conn() as conn:  # type: ignore[attr-defined]
            conn.execute(
                """
                INSERT INTO memory_short_term_messages (session_id, runtime_id, role, content, ts)
                VALUES (?, ?, ?, ?, ?)
                """,
                (session_id, runtime_id, role, content, datetime.utcnow().isoformat()),
            )

    def test_short_term_read_and_trim_are_runtime_agnostic(self) -> None:
        session_id = f"session_{uuid4().hex[:10]}"
        storage.ensure_session(session_id)

        self._insert_short_term_row(
            session_id=session_id,
            runtime_id="rt_old_a",
            role="user",
            content="m1",
        )
        self._insert_short_term_row(
            session_id=session_id,
            runtime_id="rt_old_b",
            role="assistant",
            content="m2",
        )
        self._insert_short_term_row(
            session_id=session_id,
            runtime_id="rt_new",
            role="user",
            content="m3",
        )

        rows = storage.memory_load_short_term_messages(session_id)
        self.assertEqual([r["content"] for r in rows], ["m1", "m2", "m3"])

        debug_rows = storage.memory_load_short_term_messages_for_debug(session_id, limit_n=2)
        self.assertEqual([r["content"] for r in debug_rows], ["m2", "m3"])

        deleted = storage.memory_trim_short_term_messages(session_id, keep_last=1)
        self.assertEqual(deleted, 2)

        remaining = storage.memory_load_short_term_messages(session_id)
        self.assertEqual([r["content"] for r in remaining], ["m3"])

    def test_init_db_does_not_cleanup_by_runtime_anymore(self) -> None:
        session_id = f"session_{uuid4().hex[:10]}"
        storage.ensure_session(session_id)
        marker = f"persist_{uuid4().hex[:8]}"
        self._insert_short_term_row(
            session_id=session_id,
            runtime_id="legacy_runtime",
            role="user",
            content=marker,
        )

        storage.init_db()

        rows = storage.memory_load_short_term_messages(session_id)
        self.assertTrue(any(r["content"] == marker for r in rows))


class DebugMemoryLayersRouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        storage.init_db()

    def test_debug_memory_layers_uses_explicit_session_id(self) -> None:
        user_id = f"user_{uuid4().hex[:10]}"
        active_project = storage.create_project(user_id=user_id, name=f"A_{uuid4().hex[:4]}", activate=True)
        target_project = storage.create_project(user_id=user_id, name=f"B_{uuid4().hex[:4]}", activate=False)
        active_session_id = str(active_project["session_id"])
        target_session_id = str(target_project["session_id"])
        marker = f"marker_{uuid4().hex[:8]}"

        storage.memory_append_short_term_message(
            session_id=target_session_id,
            role="user",
            content=marker,
            timestamp=datetime.utcnow().isoformat(),
        )
        storage.memory_append_short_term_message(
            session_id=target_session_id,
            role="assistant",
            content="ack",
            timestamp=datetime.utcnow().isoformat(),
        )

        with webapp.app.test_client() as client:
            client.set_cookie("user_id", user_id)
            client.set_cookie("session_id", active_session_id)
            res = client.get(f"/debug/memory-layers?session_id={target_session_id}")

        self.assertEqual(res.status_code, 200)
        payload = res.get_json() or {}
        self.assertEqual(payload.get("resolved_session_id"), target_session_id)
        short_term = payload.get("short_term") or {}
        turns = short_term.get("turns") or []
        self.assertGreaterEqual(short_term.get("turns_count", 0), 2)
        self.assertTrue(any(marker in str(turn.get("full") or "") for turn in turns))


if __name__ == "__main__":
    unittest.main()
