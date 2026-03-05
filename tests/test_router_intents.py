from __future__ import annotations

import unittest

from memory.router import MemoryRouter


class RouterIntentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.router = MemoryRouter(llm_client=None)

    def test_task_intent_detects_app_creation_goal(self) -> None:
        text = "хочу зарабатывать на ios приложениях"
        self.assertTrue(self.router._is_task_intent(text))  # noqa: SLF001

    def test_execution_guard_allows_helpful_plan_request(self) -> None:
        allowed = self.router.is_execution_allowed_message(
            text="Нужен полный план для iOS приложения с нуля до первой покупки",
            current_step="Уточнить требования",
        )
        self.assertTrue(allowed)

    def test_execution_guard_blocks_explicit_context_switch(self) -> None:
        allowed = self.router.is_execution_allowed_message(
            text="Забудь про авторизацию, давай сделаем главный экран",
            current_step="Реализовать экран авторизации",
        )
        self.assertFalse(allowed)


if __name__ == "__main__":
    unittest.main()
