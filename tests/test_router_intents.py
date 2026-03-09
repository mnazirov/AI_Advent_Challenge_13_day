from __future__ import annotations

import unittest
from uuid import uuid4

import storage
from llm.client import LLMChatResponse, LLMChoice, LLMMessage, LLMUsage
from memory.long_term import LongTermMemory
from memory.models import TaskState
from memory.router import MemoryRouter
from memory.working import WorkingMemory


class _RouterIntentMockClient:
    def chat_completion(self, *, model: str, messages: list[dict], temperature: float = 0.0, max_tokens: int = 80, **kwargs):
        del model, temperature, max_tokens, kwargs
        prompt = str(messages[0].get("content") if messages else "")
        lower = prompt.lower()

        if "intent: task_intent" in lower:
            match = "хочу зарабатывать на ios приложениях" in lower
            payload = f'{{"match": {"true" if match else "false"}, "confidence": 0.95, "reason": "mock"}}'
        elif "execution state of a task-state machine" in lower:
            if "забудь про авторизацию" in lower:
                payload = '{"allow": false, "confidence": 0.95, "reason": "context switch"}'
            elif "полный план для ios приложения" in lower:
                payload = '{"allow": true, "confidence": 0.95, "reason": "helpful"}'
            else:
                payload = '{"allow": false, "confidence": 0.95, "reason": "mock"}'
        else:
            payload = '{"match": false, "confidence": 0.95, "reason": "mock"}'

        return LLMChatResponse(
            id="mock",
            model="mock",
            choices=[LLMChoice(message=LLMMessage(content=payload), finish_reason="stop")],
            usage=LLMUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )


class _RouterInvalidPayloadClient:
    def chat_completion(self, *, model: str, messages: list[dict], temperature: float = 0.0, max_tokens: int = 80, **kwargs):
        del model, temperature, max_tokens, kwargs
        prompt = str(messages[0].get("content") if messages else "")
        lower = prompt.lower()
        if "recovery intent classifier" in lower:
            if "intent: task_intent" in lower:
                return LLMChatResponse(
                    id="mock-recovery-task",
                    model="mock-recovery",
                    choices=[
                        LLMChoice(
                            message=LLMMessage(content='{"match": true, "confidence": 0.91, "reason": "mock"}'),
                            finish_reason="stop",
                        )
                    ],
                    usage=LLMUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
                )
            if "intent: plan_formation_intent" in lower:
                return LLMChatResponse(
                    id="mock-recovery-plan",
                    model="mock-recovery",
                    choices=[
                        LLMChoice(
                            message=LLMMessage(content='{"match": true, "confidence": 0.91, "reason": "mock"}'),
                            finish_reason="stop",
                        )
                    ],
                    usage=LLMUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
                )
        return LLMChatResponse(
            id="mock-invalid",
            model="mock-invalid",
            choices=[LLMChoice(message=LLMMessage(content="not a json payload"), finish_reason="stop")],
            usage=LLMUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )


class _RouterNoRecoveryClient:
    def chat_completion(self, *, model: str, messages: list[dict], temperature: float = 0.0, max_tokens: int = 80, **kwargs):
        del model, temperature, max_tokens, kwargs, messages
        return LLMChatResponse(
            id="mock-empty",
            model="mock-empty",
            choices=[LLMChoice(message=LLMMessage(content=""), finish_reason="stop")],
            usage=LLMUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )


class _RouterRawParsedClient:
    def chat_completion(self, *, model: str, messages: list[dict], temperature: float = 0.0, max_tokens: int = 80, **kwargs):
        del model, temperature, max_tokens, kwargs
        prompt = str(messages[0].get("content") if messages else "")
        lower = prompt.lower()
        match = False
        if "intent: task_intent" in lower:
            match = "напиши расширение" in lower
        payload = {"match": match, "confidence": 0.92, "reason": "raw_parsed"}
        return LLMChatResponse(
            id="mock-raw",
            model="mock-raw",
            choices=[LLMChoice(message=LLMMessage(content=""), finish_reason="stop")],
            usage=LLMUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            raw={"choices": [{"message": {"parsed": payload}}]},
        )


class RouterIntentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        storage.init_db()

    def test_task_intent_detects_app_creation_goal(self) -> None:
        router = MemoryRouter(llm_client=_RouterIntentMockClient())
        self.assertTrue(router._is_task_intent("хочу зарабатывать на ios приложениях"))  # noqa: SLF001

    def test_execution_guard_allows_helpful_plan_request(self) -> None:
        router = MemoryRouter(llm_client=_RouterIntentMockClient())
        allowed = router.is_execution_allowed_message(
            text="Нужен полный план для iOS приложения с нуля до первой покупки",
            current_step="Уточнить требования",
        )
        self.assertTrue(allowed)

    def test_execution_policy_uses_dedicated_budget_purpose(self) -> None:
        router = MemoryRouter(llm_client=_RouterIntentMockClient())
        purposes: list[str] = []

        def reserve(purpose: str) -> bool:
            purposes.append(str(purpose))
            return True

        router.set_aux_llm_budget_reserver(reserve)
        allowed = router.is_execution_allowed_message(
            text="Нужен полный план для iOS приложения с нуля до первой покупки",
            current_step="Уточнить требования",
        )
        self.assertTrue(allowed)
        self.assertIn("execution_message_policy", purposes)
        self.assertNotIn("execution_guard", purposes)

    def test_execution_guard_blocks_explicit_context_switch(self) -> None:
        router = MemoryRouter(llm_client=_RouterIntentMockClient())
        allowed = router.is_execution_allowed_message(
            text="Забудь про авторизацию, давай сделаем главный экран",
            current_step="Реализовать экран авторизации",
        )
        self.assertFalse(allowed)

    def test_execution_fallback_allows_completion_without_llm(self) -> None:
        router = MemoryRouter(llm_client=None)
        allowed = router.is_execution_allowed_message(
            text="Шаг выполнен, двигаемся дальше",
            current_step="Собрать экран авторизации в SwiftUI",
        )
        self.assertTrue(allowed)

    def test_router_no_llm_does_not_mutate_on_free_text(self) -> None:
        router = MemoryRouter(llm_client=None)
        working = WorkingMemory()
        long_term = LongTermMemory()
        session_id = f"router_no_llm_mutation_{uuid4().hex[:8]}"
        storage.ensure_session(session_id)
        ctx = working.start_task(session_id, "Тестовая задача")
        working.update(session_id, plan=["Шаг 1"], current_step="Шаг 1")
        ctx = working.load(session_id)
        assert ctx is not None
        working.transition_state(ctx, TaskState.EXECUTION)
        working.save(ctx)

        events = router.route_user_message(
            session_id=session_id,
            user_id="u1",
            user_message="Шаг выполнен",
            working=working,
            long_term=long_term,
        )
        after = working.load(session_id)
        assert after is not None
        self.assertEqual(after.done, [])
        self.assertEqual(after.current_step, "Шаг 1")
        self.assertEqual([e.layer for e in events], [])

    def test_router_client_intent_completes_step_without_nl_parsing(self) -> None:
        router = MemoryRouter(llm_client=None)
        working = WorkingMemory()
        long_term = LongTermMemory()
        session_id = f"router_structured_completion_{uuid4().hex[:8]}"
        storage.ensure_session(session_id)
        ctx = working.start_task(session_id, "Тестовая задача")
        working.update(session_id, plan=["Шаг 1"], current_step="Шаг 1")
        ctx = working.load(session_id)
        assert ctx is not None
        working.transition_state(ctx, TaskState.EXECUTION)
        working.save(ctx)

        router.route_user_message(
            session_id=session_id,
            user_id="u1",
            user_message="ok",
            working=working,
            long_term=long_term,
            client_intent={"intent": "step_completed"},
        )
        after = working.load(session_id)
        assert after is not None
        self.assertEqual(after.done, ["Шаг 1"])
        self.assertEqual(after.state, TaskState.VALIDATION)

    def test_router_budget_exhausted_keeps_state_stable(self) -> None:
        router = MemoryRouter(llm_client=_RouterIntentMockClient())
        router.set_aux_llm_budget_reserver(lambda _purpose: False)
        working = WorkingMemory()
        long_term = LongTermMemory()
        session_id = f"router_budget_stable_{uuid4().hex[:8]}"
        storage.ensure_session(session_id)
        ctx = working.start_task(session_id, "Тестовая задача")
        working.update(session_id, plan=["Шаг 1"], current_step="Шаг 1")
        ctx = working.load(session_id)
        assert ctx is not None
        working.transition_state(ctx, TaskState.EXECUTION)
        working.save(ctx)

        router.route_user_message(
            session_id=session_id,
            user_id="u1",
            user_message="Шаг выполнен",
            working=working,
            long_term=long_term,
        )
        after = working.load(session_id)
        assert after is not None
        self.assertEqual(after.done, [])
        self.assertEqual(after.current_step, "Шаг 1")

    def test_task_intent_recovery_analysis_handles_invalid_payload(self) -> None:
        router = MemoryRouter(llm_client=_RouterInvalidPayloadClient())
        decision = router._task_intent(text="Напиши расширение для безопасного доступа к массиву")  # noqa: SLF001
        self.assertTrue(decision.is_match)
        self.assertEqual(decision.reason_code, "intent_match_recovery_analysis")

    def test_plan_intent_recovery_analysis_handles_invalid_payload(self) -> None:
        router = MemoryRouter(llm_client=_RouterInvalidPayloadClient())
        decision = router._plan_formation_intent(text="Сформируй план реализации фичи")  # noqa: SLF001
        self.assertTrue(decision.is_match)
        self.assertEqual(decision.reason_code, "intent_match_recovery_analysis")

    def test_route_starts_task_on_recovery_analysis_when_intent_payload_invalid(self) -> None:
        router = MemoryRouter(llm_client=_RouterInvalidPayloadClient())
        working = WorkingMemory()
        long_term = LongTermMemory()
        session_id = f"router_recovery_analysis_start_{uuid4().hex[:8]}"
        storage.ensure_session(session_id)

        events = router.route_user_message(
            session_id=session_id,
            user_id="u1",
            user_message="Напиши расширение для безопасного доступа к массиву",
            working=working,
            long_term=long_term,
        )
        ctx = working.load(session_id)
        assert ctx is not None
        self.assertEqual(ctx.state, TaskState.PLANNING)
        self.assertIn("Напиши расширение", ctx.task)
        self.assertTrue(any(e.layer == "working" for e in events))

    def test_route_starts_task_on_bootstrap_heuristic_when_llm_payloads_are_empty(self) -> None:
        router = MemoryRouter(llm_client=_RouterNoRecoveryClient())
        working = WorkingMemory()
        long_term = LongTermMemory()
        session_id = f"router_bootstrap_heuristic_{uuid4().hex[:8]}"
        storage.ensure_session(session_id)

        events = router.route_user_message(
            session_id=session_id,
            user_id="u1",
            user_message="Напиши расширение для безопастному обращению к массивам",
            working=working,
            long_term=long_term,
        )
        ctx = working.load(session_id)
        assert ctx is not None
        self.assertEqual(ctx.state, TaskState.PLANNING)
        self.assertIn("Напиши расширение", ctx.task)
        self.assertTrue(any(e.layer == "working" for e in events))

    def test_extract_first_json_object_accepts_python_style_dict(self) -> None:
        router = MemoryRouter(llm_client=None)
        payload = router._extract_first_json_object("{'match': True, 'confidence': 0.91, 'reason': 'mock'}")  # noqa: SLF001
        self.assertIsInstance(payload, dict)
        assert payload is not None
        self.assertEqual(payload.get("match"), True)
        self.assertEqual(payload.get("confidence"), 0.91)

    def test_route_starts_task_when_intent_payload_is_in_raw_response(self) -> None:
        router = MemoryRouter(llm_client=_RouterRawParsedClient())
        working = WorkingMemory()
        long_term = LongTermMemory()
        session_id = f"router_raw_payload_start_{uuid4().hex[:8]}"
        storage.ensure_session(session_id)

        router.route_user_message(
            session_id=session_id,
            user_id="u1",
            user_message="Напиши расширение для безопасного доступа к массиву",
            working=working,
            long_term=long_term,
        )
        ctx = working.load(session_id)
        assert ctx is not None
        self.assertEqual(ctx.state, TaskState.PLANNING)
        self.assertIn("Напиши расширение", ctx.task)


if __name__ == "__main__":
    unittest.main()
