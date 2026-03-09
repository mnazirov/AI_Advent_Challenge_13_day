from __future__ import annotations

import unittest
from datetime import datetime
from uuid import uuid4

import app as webapp
import storage
from memory.manager import MemoryManager


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

    def test_debug_memory_profile_returns_canonical_fields(self) -> None:
        user_id = f"user_{uuid4().hex[:10]}"
        project = storage.create_project(user_id=user_id, name=f"P_{uuid4().hex[:4]}", activate=True)
        session_id = str(project["session_id"])
        webapp.agent.memory.long_term.update_profile_field(
            user_id=user_id,
            field="user_role_level",
            value="новичок",
            source="user_explicit",
            verified=True,
        )
        webapp.agent.memory.long_term.update_profile_field(
            user_id=user_id,
            field="stack_tools",
            value=["Swift", "SwiftUI"],
            source="user_explicit",
            verified=True,
        )

        with webapp.app.test_client() as client:
            client.set_cookie("user_id", user_id)
            client.set_cookie("session_id", session_id)
            res = client.get("/debug/memory/profile")

        self.assertEqual(res.status_code, 200)
        payload = res.get_json() or {}
        profile = payload.get("profile") or {}
        self.assertEqual((profile.get("user_role_level") or {}).get("value"), "новичок")
        self.assertIn("Swift", (profile.get("stack_tools") or {}).get("value") or [])

    def test_debug_clear_short_term_route_only_clears_short_term(self) -> None:
        user_id = f"user_{uuid4().hex[:10]}"
        project = storage.create_project(user_id=user_id, name=f"S_{uuid4().hex[:4]}", activate=True)
        session_id = str(project["session_id"])

        webapp.agent.memory.append_turn(session_id=session_id, user_message="u1", assistant_message="a1")
        webapp.agent.memory.working.start_task(session_id=session_id, goal="Собрать onboarding")
        webapp.agent.memory.long_term.add_note(user_id=user_id, text="Запомнить paywall", source="user")
        storage.save_message(session_id, "user", "История 1")
        storage.save_message(session_id, "assistant", "История 2")

        with webapp.app.test_client() as client:
            client.set_cookie("user_id", user_id)
            client.set_cookie("session_id", session_id)
            res = client.post("/debug/memory/short-term/clear", json={"session_id": session_id})

        self.assertEqual(res.status_code, 200)
        payload = res.get_json() or {}
        snapshot = payload.get("snapshot") or {}
        self.assertTrue(payload.get("cleared"))
        self.assertEqual((snapshot.get("short_term") or {}).get("turns_count"), 0)
        self.assertTrue((snapshot.get("working") or {}).get("present"))
        self.assertEqual(len((snapshot.get("long_term") or {}).get("notes_top_k") or []), 1)
        restored = storage.load_session(session_id) or {}
        self.assertEqual(restored.get("messages") or [], [])

    def test_debug_clear_working_route_only_clears_working(self) -> None:
        user_id = f"user_{uuid4().hex[:10]}"
        project = storage.create_project(user_id=user_id, name=f"W_{uuid4().hex[:4]}", activate=True)
        session_id = str(project["session_id"])

        webapp.agent.memory.append_turn(session_id=session_id, user_message="u1", assistant_message="a1")
        webapp.agent.memory.working.start_task(session_id=session_id, goal="Проверить execution flow")
        webapp.agent.memory.long_term.add_note(user_id=user_id, text="Важно сохранить профиль", source="user")
        storage.save_message(session_id, "user", "История 1")
        storage.save_message(session_id, "assistant", "История 2")

        with webapp.app.test_client() as client:
            client.set_cookie("user_id", user_id)
            client.set_cookie("session_id", session_id)
            res = client.post("/debug/memory/working/clear", json={"session_id": session_id})

        self.assertEqual(res.status_code, 200)
        payload = res.get_json() or {}
        snapshot = payload.get("snapshot") or {}
        self.assertTrue(payload.get("cleared"))
        self.assertFalse((snapshot.get("working") or {}).get("present"))
        self.assertEqual((snapshot.get("short_term") or {}).get("turns_count"), 2)
        self.assertEqual(len((snapshot.get("long_term") or {}).get("notes_top_k") or []), 1)
        restored = storage.load_session(session_id) or {}
        self.assertEqual(len(restored.get("messages") or []), 2)

    def test_debug_clear_long_term_route_only_clears_long_term(self) -> None:
        user_id = f"user_{uuid4().hex[:10]}"
        project = storage.create_project(user_id=user_id, name=f"L_{uuid4().hex[:4]}", activate=True)
        session_id = str(project["session_id"])

        webapp.agent.memory.append_turn(session_id=session_id, user_message="u1", assistant_message="a1")
        webapp.agent.memory.working.start_task(session_id=session_id, goal="Проверить long-term clear")
        storage.save_message(session_id, "user", "История 1")
        storage.save_message(session_id, "assistant", "История 2")
        webapp.agent.memory.long_term.update_profile_field(
            user_id=user_id,
            field="response_style",
            value="Кратко и по делу",
            source="user_explicit",
            verified=True,
        )
        webapp.agent.memory.long_term.add_decision(user_id=user_id, text="Используем StoreKit 2", source="user")
        webapp.agent.memory.long_term.add_note(user_id=user_id, text="MVP с подпиской", source="user")
        storage.memory_add_longterm_pending(
            user_id=user_id,
            entry_type="note",
            text="pending note",
            tags=[],
            source="assistant",
        )

        with webapp.app.test_client() as client:
            client.set_cookie("user_id", user_id)
            client.set_cookie("session_id", session_id)
            res = client.post("/debug/memory/long-term/clear", json={"session_id": session_id})

        self.assertEqual(res.status_code, 200)
        payload = res.get_json() or {}
        snapshot = payload.get("snapshot") or {}
        self.assertTrue(payload.get("cleared"))
        self.assertEqual(len((snapshot.get("long_term") or {}).get("decisions_top_k") or []), 0)
        self.assertEqual(len((snapshot.get("long_term") or {}).get("notes_top_k") or []), 0)
        self.assertFalse((snapshot.get("long_term") or {}).get("has_profile_data"))
        self.assertEqual((snapshot.get("short_term") or {}).get("turns_count"), 2)
        self.assertTrue((snapshot.get("working") or {}).get("present"))
        restored = storage.load_session(session_id) or {}
        self.assertEqual(len(restored.get("messages") or []), 2)

    def test_debug_clear_short_term_resets_runtime_conversation_for_active_session(self) -> None:
        user_id = f"user_{uuid4().hex[:10]}"
        project = storage.create_project(user_id=user_id, name=f"R_{uuid4().hex[:4]}", activate=True)
        session_id = str(project["session_id"])
        previous_runtime = getattr(webapp, "_runtime_session_id", None)
        previous_history = list(webapp.agent.conversation_history)
        try:
            webapp._runtime_session_id = session_id
            webapp.agent.conversation_history = [
                {"role": "user", "content": "Старый вопрос"},
                {"role": "assistant", "content": "Старый ответ"},
            ]
            storage.save_message(session_id, "user", "Старый вопрос")
            storage.save_message(session_id, "assistant", "Старый ответ")

            with webapp.app.test_client() as client:
                client.set_cookie("user_id", user_id)
                client.set_cookie("session_id", session_id)
                res = client.post("/debug/memory/short-term/clear", json={"session_id": session_id})

            self.assertEqual(res.status_code, 200)
            self.assertEqual(webapp.agent.conversation_history, [])
        finally:
            webapp._runtime_session_id = previous_runtime
            webapp.agent.conversation_history = previous_history

class MemoryLayerClearManagerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        storage.init_db()

    def setUp(self) -> None:
        self.memory = MemoryManager()
        self.user_id = f"user_{uuid4().hex[:10]}"
        self.session_id = f"session_{uuid4().hex[:10]}"
        storage.ensure_session(self.session_id)

    def test_clear_short_term_layer_is_isolated(self) -> None:
        self.memory.append_turn(session_id=self.session_id, user_message="u1", assistant_message="a1")
        self.memory.working.start_task(session_id=self.session_id, goal="Собрать onboarding")
        self.memory.long_term.add_note(user_id=self.user_id, text="Запомнить решение", source="user")
        storage.save_message(self.session_id, "user", "История 1")
        storage.save_message(self.session_id, "assistant", "История 2")

        cleared = self.memory.clear_short_term_layer(session_id=self.session_id)

        self.assertTrue(cleared)
        self.assertEqual(len(self.memory.short_term.get_context(self.session_id)), 0)
        self.assertIsNotNone(self.memory.working.load(self.session_id))
        self.assertEqual(len(self.memory.long_term.retrieve(user_id=self.user_id, query="", top_k=3).get("notes") or []), 1)
        self.assertEqual((storage.load_session(self.session_id) or {}).get("messages") or [], [])
        last_event = self.memory.get_recent_write_events(session_id=self.session_id, limit=1)[0]
        self.assertEqual(last_event.get("layer"), "short_term")
        self.assertEqual(last_event.get("operation"), "clear")

    def test_clear_long_term_layer_is_isolated(self) -> None:
        self.memory.append_turn(session_id=self.session_id, user_message="u1", assistant_message="a1")
        self.memory.working.start_task(session_id=self.session_id, goal="Собрать onboarding")
        self.memory.long_term.update_profile_field(
            user_id=self.user_id,
            field="response_style",
            value="Кратко",
            source="user_explicit",
            verified=True,
        )
        self.memory.long_term.add_decision(user_id=self.user_id, text="Используем UIKit", source="user")
        self.memory.long_term.add_note(user_id=self.user_id, text="Сделать paywall", source="user")
        storage.memory_add_longterm_pending(
            user_id=self.user_id,
            entry_type="note",
            text="pending",
            tags=[],
            source="assistant",
        )

        cleared = self.memory.clear_long_term_layer(session_id=self.session_id, user_id=self.user_id)

        self.assertTrue(cleared)
        self.assertEqual(len(self.memory.short_term.get_context(self.session_id)), 2)
        self.assertIsNotNone(self.memory.working.load(self.session_id))
        self.assertIsNone(storage.memory_load_longterm_profile(self.user_id))
        self.assertEqual(len(storage.memory_list_longterm_decisions(self.user_id)), 0)
        self.assertEqual(len(storage.memory_list_longterm_notes(self.user_id)), 0)
        self.assertEqual(len(storage.memory_list_longterm_pending(self.user_id)), 0)
        last_event = self.memory.get_recent_write_events(session_id=self.session_id, limit=1)[0]
        self.assertEqual(last_event.get("layer"), "long_term")
        self.assertEqual(last_event.get("operation"), "clear")

    def test_repeat_clear_empty_layers_returns_false(self) -> None:
        self.assertFalse(self.memory.clear_short_term_layer(session_id=self.session_id))
        self.assertFalse(self.memory.clear_working_layer(session_id=self.session_id))
        self.assertFalse(self.memory.clear_long_term_layer(session_id=self.session_id, user_id=self.user_id))


if __name__ == "__main__":
    unittest.main()
