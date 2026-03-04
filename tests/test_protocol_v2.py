from __future__ import annotations

import unittest
from uuid import uuid4

import storage
from agent import IOSAgent
from memory.manager import MemoryManager


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
