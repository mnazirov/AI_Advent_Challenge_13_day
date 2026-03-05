from __future__ import annotations

import unittest
from uuid import uuid4

import storage
from agent import IOSAgent
from memory.manager import MemoryManager
from memory.models import ProfileSource
from memory.protocol import PROTOCOL_PROFILE_KEY


class ProtocolV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        storage.init_db()

    def setUp(self) -> None:
        self.memory = MemoryManager()
        self.user_id = f"user_{uuid4().hex[:8]}"
        self.session_id = f"session_{uuid4().hex[:8]}"
        storage.ensure_session(self.session_id)
        self.memory.ensure_protocol_profile(user_id=self.user_id)

    def test_profile_is_collected_organically(self):
        runtime = self.memory.prepare_protocol_turn(
            session_id=self.session_id,
            user_id=self.user_id,
            user_message="Я новичок. Хочу сделать приложение для бегунов с подпиской.",
        )
        profile = (runtime.get("protocol_state") or {}).get("profile") or {}
        self.assertEqual(profile.get("experience_level"), "новичок")
        self.assertEqual(profile.get("monetization_model"), "подписка")
        self.assertIn("бегун", str(profile.get("app_idea", "")).lower())
        self.assertIn((runtime.get("protocol_state") or {}).get("phase"), {"ONBOARDING", "PLANNING"})

    def test_phase_transitions_are_automatic(self):
        self.memory.prepare_protocol_turn(
            session_id=self.session_id,
            user_id=self.user_id,
            user_message="Я делаю приложение для изучения слов, аудитория — студенты.",
        )
        planning = self.memory.prepare_protocol_turn(
            session_id=self.session_id,
            user_id=self.user_id,
            user_message="MVP согласован, план готов.",
        )
        phase_after_planning = (planning.get("protocol_state") or {}).get("phase")
        self.assertIn(phase_after_planning, {"PLANNING", "EXECUTION"})

        execution = self.memory.prepare_protocol_turn(
            session_id=self.session_id,
            user_id=self.user_id,
            user_message="Давай писать код экрана и собрать рабочий flow в SwiftUI.",
        )
        self.assertEqual((execution.get("protocol_state") or {}).get("phase"), "EXECUTION")

        monetization = self.memory.prepare_protocol_turn(
            session_id=self.session_id,
            user_id=self.user_id,
            user_message="Теперь подключим StoreKit2 и настроим paywall.",
        )
        self.assertEqual((monetization.get("protocol_state") or {}).get("phase"), "MONETIZATION")

    def test_invariants_autofix_monetization_model(self):
        self.memory.prepare_protocol_turn(
            session_id=self.session_id,
            user_id=self.user_id,
            user_message="К монетизации: давай сразу делать paywall.",
        )
        runtime = self.memory.prepare_protocol_turn(
            session_id=self.session_id,
            user_id=self.user_id,
            user_message="Начнем с тестового экрана оплаты.",
        )
        profile = (runtime.get("protocol_state") or {}).get("profile") or {}
        self.assertTrue(bool(profile.get("monetization_model")))
        report = runtime.get("invariant_report") or {}
        self.assertIn("overall_status", report)

    def test_canonical_profile_sync_is_applied_after_protocol_turn(self):
        runtime = self.memory.prepare_protocol_turn(
            session_id=self.session_id,
            user_id=self.user_id,
            user_message=(
                "Я новичок. Делаю приложение для бегунов. "
                "Целевая аудитория: начинающие бегуны. "
                "Стек SwiftUI. Монетизация: подписка."
            ),
        )
        profile = self.memory.get_profile_snapshot(user_id=self.user_id, session_id=self.session_id)
        stack_tools = (profile.get("stack_tools") or {}).get("value") or []
        hard_constraints = (profile.get("hard_constraints") or {}).get("value") or []
        project_context = (profile.get("project_context") or {}).get("value") or {}
        sync_meta = runtime.get("canonical_sync") or {}

        self.assertEqual((profile.get("user_role_level") or {}).get("value"), "новичок")
        self.assertIn("Swift", stack_tools)
        self.assertIn("SwiftUI", stack_tools)
        self.assertTrue(any(str(item).startswith("Монетизация:") for item in hard_constraints))
        self.assertTrue(any(str(item).startswith("ЦА:") for item in hard_constraints))
        self.assertTrue(bool(str(project_context.get("project_name") or "").strip()))
        self.assertEqual(sync_meta.get("status"), "updated")
        self.assertIn("user_role_level", sync_meta.get("updated_fields") or [])

    def test_verified_field_is_not_overwritten_and_conflict_is_recorded(self):
        self.memory.long_term.update_profile_field(
            user_id=self.user_id,
            field="user_role_level",
            value="продвинутый",
            source=ProfileSource.USER_EXPLICIT,
            verified=True,
        )
        runtime = self.memory.prepare_protocol_turn(
            session_id=self.session_id,
            user_id=self.user_id,
            user_message="Я новичок. Хочу приложение для бегунов.",
        )
        profile = self.memory.get_profile_snapshot(user_id=self.user_id, session_id=self.session_id)
        conflicts = profile.get("conflicts") or []
        sync_meta = runtime.get("canonical_sync") or {}

        self.assertEqual((profile.get("user_role_level") or {}).get("value"), "продвинутый")
        self.assertTrue(any(str(item.get("field") or "") == "user_role_level" for item in conflicts if isinstance(item, dict)))
        self.assertIn("user_role_level", sync_meta.get("conflict_fields") or [])

    def test_repeated_sync_is_idempotent(self):
        first = self.memory.prepare_protocol_turn(
            session_id=self.session_id,
            user_id=self.user_id,
            user_message="Я новичок. Делаю приложение для бегунов с подпиской на SwiftUI.",
        )
        first_profile = self.memory.get_profile_snapshot(user_id=self.user_id, session_id=self.session_id)
        first_role_updated = str((first_profile.get("user_role_level") or {}).get("updated_at") or "")
        protocol_sync_events_before = [
            e
            for e in self.memory.get_recent_write_events(session_id=self.session_id, limit=20)
            if str(e.get("source") or "") == "protocol_sync"
        ]

        second = self.memory.prepare_protocol_turn(
            session_id=self.session_id,
            user_id=self.user_id,
            user_message="Ок, продолжаем.",
        )
        second_profile = self.memory.get_profile_snapshot(user_id=self.user_id, session_id=self.session_id)
        second_role_updated = str((second_profile.get("user_role_level") or {}).get("updated_at") or "")
        protocol_sync_events_after = [
            e
            for e in self.memory.get_recent_write_events(session_id=self.session_id, limit=20)
            if str(e.get("source") or "") == "protocol_sync"
        ]

        self.assertEqual(first.get("canonical_sync", {}).get("status"), "updated")
        self.assertEqual(second.get("canonical_sync", {}).get("status"), "no_changes")
        self.assertEqual(first_role_updated, second_role_updated)
        self.assertEqual(len(protocol_sync_events_before), len(protocol_sync_events_after))

    def test_lazy_backfill_syncs_existing_protocol_profile_on_snapshot(self):
        self.memory.long_term.add_profile_extra_field(
            user_id=self.user_id,
            field=PROTOCOL_PROFILE_KEY,
            value={
                "experience_level": "новичок",
                "app_idea": "Приложение для изучения слов",
                "target_audience": "студенты",
                "stack": "Swift + SwiftUI",
                "monetization_model": "подписка",
                "current_progress": "Собран экран онбординга",
                "updated_at": "2026-03-05T00:00:00",
            },
            source=ProfileSource.USER_EXPLICIT,
        )
        profile = self.memory.get_profile_snapshot(user_id=self.user_id, session_id=self.session_id)
        hard_constraints = (profile.get("hard_constraints") or {}).get("value") or []
        project_context = (profile.get("project_context") or {}).get("value") or {}
        key_decisions = project_context.get("key_decisions") or []

        self.assertEqual((profile.get("user_role_level") or {}).get("value"), "новичок")
        self.assertIn("Swift", (profile.get("stack_tools") or {}).get("value") or [])
        self.assertTrue(any(str(item).startswith("Монетизация:") for item in hard_constraints))
        self.assertTrue(any(str(item).startswith("Текущий прогресс:") for item in key_decisions))

    def test_internal_tags_are_removed_from_user_text(self):
        agent = IOSAgent.__new__(IOSAgent)
        internal, external = agent._split_internal_external(
            "<internal>phase=PLANNING; check=ok</internal><external>Привет! Начинаем.</external>"
        )
        cleaned = agent._strip_internal_artifacts(external)
        self.assertIn("phase=PLANNING", internal)
        self.assertEqual(cleaned, "Привет! Начинаем.")
        self.assertNotIn("<internal>", cleaned)
        self.assertNotIn("INV:", cleaned)
        self.assertNotIn("NEXT:", cleaned)


if __name__ == "__main__":
    unittest.main()
