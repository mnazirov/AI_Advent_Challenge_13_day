from __future__ import annotations

import unittest
from uuid import uuid4

import storage
from agent import IOSAgent
from memory.manager import MemoryManager
from memory.models import TaskState


class _DummyContextStrategy:
    active = "none"

    @staticmethod
    def stats(_history: list[dict[str, str]]) -> dict:
        return {}


class ValidationDoneFlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        storage.init_db()

    def setUp(self) -> None:
        self.memory = MemoryManager()
        self.user_id = f"user_{uuid4().hex[:8]}"
        self.session_id = f"session_{uuid4().hex[:8]}"
        storage.ensure_session(self.session_id)
        self.memory.ensure_protocol_profile(user_id=self.user_id)

    def _prepare_execution_task(self, *, steps: list[str]) -> None:
        self.memory.working.start_task(self.session_id, "Тестовая задача")
        self.memory.working.update(self.session_id, plan=steps, current_step=steps[0])
        ctx = self.memory.working.load(self.session_id)
        assert ctx is not None
        self.memory.working.transition_state(ctx, TaskState.EXECUTION)
        self.memory.working.save(ctx)

    def test_last_execution_step_moves_to_validation(self) -> None:
        self._prepare_execution_task(steps=["Шаг 1", "Шаг 2"])
        self.memory.route_user_message(
            session_id=self.session_id,
            user_id=self.user_id,
            user_message="Шаг выполнен",
        )
        self.memory.route_user_message(
            session_id=self.session_id,
            user_id=self.user_id,
            user_message="Шаг выполнен",
        )
        ctx = self.memory.working.load(self.session_id)
        assert ctx is not None
        self.assertEqual(ctx.state, TaskState.VALIDATION)
        self.assertEqual(ctx.done, ["Шаг 1", "Шаг 2"])

    def test_validation_flow_supports_checklist_reject_and_auto_done(self) -> None:
        self._prepare_execution_task(steps=["Шаг 1"])
        self.memory.route_user_message(
            session_id=self.session_id,
            user_id=self.user_id,
            user_message="Шаг выполнен",
        )
        ctx = self.memory.working.load(self.session_id)
        assert ctx is not None
        self.assertEqual(ctx.state, TaskState.VALIDATION)

        checklist_reply = self.memory.enforce_planning_gate(
            session_id=self.session_id,
            user_message="Покажи чеклист",
            user_id=self.user_id,
        )
        self.assertIsNone(checklist_reply)
        checklist_ctx = self.memory.working.load(self.session_id)
        assert checklist_ctx is not None
        self.assertEqual(checklist_ctx.state, TaskState.VALIDATION)

        reject_reply = self.memory.enforce_planning_gate(
            session_id=self.session_id,
            user_message="Есть замечания, нужно доработать",
            user_id=self.user_id,
        )
        self.assertIsNone(reject_reply)
        reject_ctx = self.memory.working.load(self.session_id)
        assert reject_ctx is not None
        self.assertEqual(reject_ctx.state, TaskState.EXECUTION)

        self._prepare_execution_task(steps=["Шаг 1"])
        self.memory.route_user_message(
            session_id=self.session_id,
            user_id=self.user_id,
            user_message="Шаг выполнен",
        )
        neutral_reply = self.memory.enforce_planning_gate(
            session_id=self.session_id,
            user_message="Ок, продолжаем",
            user_id=self.user_id,
        )
        self.assertEqual(neutral_reply, self.memory.VALIDATION_CONFIRMED_SIGNAL)

        actions = self.memory.get_working_actions(session_id=self.session_id)
        self.assertEqual(actions, [])

    def test_agent_autosends_validation_and_done_summaries(self) -> None:
        self._prepare_execution_task(steps=["Шаг 1"])
        self.memory.working.update(
            self.session_id,
            artifacts=[{"step": "Шаг 1", "type": "response", "ref": "report.md"}],
        )

        agent = IOSAgent.__new__(IOSAgent)
        agent.memory = self.memory
        agent.ctx = _DummyContextStrategy()
        agent.conversation_history = []
        agent.last_token_stats = None
        agent.last_memory_stats = None
        agent.last_prompt_preview = None
        agent.last_chat_response_meta = None
        agent.model = IOSAgent.DEFAULT_MODEL

        validation_reply = agent.chat(
            "Шаг выполнен",
            session_id=self.session_id,
            user_id=self.user_id,
        )
        self.assertIn("Итог выполненных шагов", validation_reply)
        self.assertIn("✓ Шаг 1", validation_reply)

        validation_ctx = self.memory.working.load(self.session_id)
        assert validation_ctx is not None
        self.assertEqual(validation_ctx.state, TaskState.VALIDATION)

        done_reply = agent.chat(
            "Ок",
            session_id=self.session_id,
            user_id=self.user_id,
        )
        self.assertIn("## Задача завершена", done_reply)
        self.assertIn("### Выполненные шаги", done_reply)
        self.assertIn("### Артефакты", done_reply)
        self.assertIn("### Архитектурные решения", done_reply)

        done_ctx = self.memory.working.load(self.session_id)
        assert done_ctx is not None
        self.assertEqual(done_ctx.state, TaskState.DONE)

        actions = self.memory.get_working_actions(session_id=self.session_id)
        self.assertEqual(actions, [])


if __name__ == "__main__":
    unittest.main()
