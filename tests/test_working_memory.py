from __future__ import annotations

import unittest
from uuid import uuid4

import storage
from memory.models import TaskState
from memory.working import WorkingMemory


class WorkingMemoryTransitionsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        storage.init_db()

    def setUp(self) -> None:
        self.working = WorkingMemory()
        self.session_id = f"session_{uuid4().hex[:8]}"
        storage.ensure_session(self.session_id)

    def _prepare_validation_state(self) -> None:
        self.working.start_task(self.session_id, "Тестовая задача")
        self.working.update(self.session_id, plan=["Шаг 1", "Шаг 2"], current_step="Шаг 1")
        ctx = self.working.load(self.session_id)
        assert ctx is not None

        self.working.transition_state(ctx, TaskState.EXECUTION)
        self.working.save(ctx)

        self.working.complete_current_step(self.session_id)
        self.working.complete_current_step(self.session_id)
        self.working.request_validation(self.session_id)

    def test_validation_to_execution_reopens_last_step(self) -> None:
        self._prepare_validation_state()
        ctx = self.working.load(self.session_id)
        assert ctx is not None
        self.assertEqual(ctx.state, TaskState.VALIDATION)

        self.working.transition_state(ctx, TaskState.EXECUTION)
        self.working.save(ctx)

        rolled_back = self.working.load(self.session_id)
        assert rolled_back is not None
        self.assertEqual(rolled_back.state, TaskState.EXECUTION)
        self.assertEqual(rolled_back.current_step, "Шаг 2")
        self.assertEqual(rolled_back.done, ["Шаг 1"])

        self.working.complete_current_step(self.session_id)
        completed = self.working.load(self.session_id)
        assert completed is not None
        self.assertEqual(completed.done, ["Шаг 1", "Шаг 2"])
        self.assertIsNone(completed.current_step)

    def test_load_does_not_repair_current_step_in_validation(self) -> None:
        self._prepare_validation_state()
        ctx = self.working.load(self.session_id)
        assert ctx is not None
        self.assertEqual(ctx.state, TaskState.VALIDATION)
        self.assertIsNone(ctx.current_step)

    def test_execution_to_done_is_forbidden(self) -> None:
        self.working.start_task(self.session_id, "Тестовая задача")
        self.working.update(self.session_id, plan=["Шаг 1"], current_step="Шаг 1")
        ctx = self.working.load(self.session_id)
        assert ctx is not None
        self.working.transition_state(ctx, TaskState.EXECUTION)
        self.working.save(ctx)
        self.working.complete_current_step(self.session_id)
        execution_ctx = self.working.load(self.session_id)
        assert execution_ctx is not None
        self.assertEqual(execution_ctx.state, TaskState.EXECUTION)
        with self.assertRaises(ValueError):
            self.working.transition_state(execution_ctx, TaskState.DONE)


if __name__ == "__main__":
    unittest.main()
