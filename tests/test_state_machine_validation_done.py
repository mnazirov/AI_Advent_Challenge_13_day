from __future__ import annotations

import re
import unittest
from uuid import uuid4

import storage
from agent import IOSAgent
from llm.client import LLMChatResponse, LLMChoice, LLMMessage, LLMUsage
from memory.intents import IntentName
from memory.manager import MemoryManager
from memory.models import TaskState


class _DummyContextStrategy:
    active = "none"

    @staticmethod
    def stats(_history: list[dict[str, str]]) -> dict:
        return {}


class _IntentMockClient:
    def chat_completion(self, **kwargs):
        messages = kwargs.get("messages") or []
        prompt = str((messages[-1] or {}).get("content") or "")
        lower = prompt.lower()
        if "ответь строго одним словом" in lower and "сообщение пользователя:" in lower:
            message_match = re.search(
                r'сообщение пользователя:\s*"(.*?)"',
                prompt,
                flags=re.IGNORECASE | re.DOTALL,
            )
            user_message = str(message_match.group(1) if message_match else "").strip().lower()
            confirmed = user_message in {"да", "давай", "подходит", "окей", "ok", "yes", "поехали", "норм"}
            rejected = (
                "?" in user_message
                or user_message.startswith("а можно")
                or user_message.startswith("нет")
                or user_message.startswith("лучше")
            )
            payload = "YES" if confirmed and not rejected else "NO"
            return LLMChatResponse(
                id="mock-confirmation",
                model="mock-confirmation",
                choices=[LLMChoice(message=LLMMessage(content=payload), finish_reason="stop")],
                usage=LLMUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            )
        if "extract canonical technology stack/framework from text." in lower:
            if "react native" in lower:
                payload = '{"stack_id": "REACT_NATIVE", "stack_label": "React Native", "confidence": 0.95}'
            elif "swiftui" in lower:
                payload = '{"stack_id": "SWIFTUI", "stack_label": "SwiftUI", "confidence": 0.95}'
            elif "swiftdata" in lower:
                payload = '{"stack_id": "SWIFTDATA", "stack_label": "SwiftData", "confidence": 0.95}'
            elif "uikit" in lower:
                payload = '{"stack_id": "UIKIT", "stack_label": "UIKit", "confidence": 0.95}'
            else:
                payload = '{"stack_id": "", "stack_label": "", "confidence": 0.95}'
            return LLMChatResponse(
                id="mock-stack",
                model="mock-stack",
                choices=[LLMChoice(message=LLMMessage(content=payload), finish_reason="stop")],
                usage=LLMUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            )
        if "extract working-memory task updates" in lower:
            current_step_match = re.search(r"^- current_step=(.*)$", prompt, flags=re.IGNORECASE | re.MULTILINE)
            current_step = str(current_step_match.group(1) if current_step_match else "").strip()
            user_message_match = re.search(r"User message:\s*(.*)$", prompt, flags=re.IGNORECASE | re.DOTALL)
            user_message = str(user_message_match.group(1) if user_message_match else "").strip().lower()
            if "шаг выполнен" in user_message and current_step:
                payload = (
                    '{"is_working_update": true, "task": "", "plan": [], "plan_steps_to_add": [], '
                    f'"current_step": "", "done_steps_to_add": ["{current_step}"], '
                    '"requirements_to_add": [], "artifacts_to_add": [], "confidence": 0.95}'
                )
            elif "сформируй план" in user_message:
                payload = (
                    '{"is_working_update": true, "task": "Тестовая задача", '
                    '"plan": ["Шаг 1", "Шаг 2"], "plan_steps_to_add": [], "current_step": "Шаг 1", '
                    '"done_steps_to_add": [], "requirements_to_add": [], "artifacts_to_add": [], "confidence": 0.95}'
                )
            else:
                payload = (
                    '{"is_working_update": false, "task": "", "plan": [], "plan_steps_to_add": [], '
                    '"current_step": "", "done_steps_to_add": [], "requirements_to_add": [], '
                    '"artifacts_to_add": [], "confidence": 0.95}'
                )
            return LLMChatResponse(
                id="mock",
                model="mock",
                choices=[LLMChoice(message=LLMMessage(content=payload), finish_reason="stop")],
                usage=LLMUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            )
        intent = ""
        message = ""
        for line in prompt.splitlines():
            if line.lower().startswith("intent:"):
                intent = line.split(":", 1)[1].strip().lower()
            if line.lower().startswith("user message:"):
                message = line.split(":", 1)[1].strip().lower()

        match = False
        if intent == "start_execution":
            match = "поехали" in message
        elif intent == "plan_approved":
            match = "поехали" in message or "утверж" in message
        elif intent == "plan_formation":
            match = "сформируй план" in message
        elif intent == "skip_mandatory_planning":
            match = "не задавай вопросов" in message or "сразу пиши код" in message
        elif intent == "goal_clarification":
            match = "уточ" in message and "цель" in message
        elif intent == "direct_code_request":
            match = "код" in message and ("сразу" in message or "готов" in message or "напиши" in message)
        elif intent == "validation_request":
            match = "валидац" in message or "провер" in message
        elif intent == "validation_checklist_request":
            match = "чеклист" in message or "checklist" in message
        elif intent == "validation_confirm":
            match = "подтверждаю завершение" in message
        elif intent == "validation_reject":
            match = "доработ" in message or "замечани" in message
        elif intent == "validation_skip_request":
            match = "skip validation" in message
        elif intent == "yes_confirmation":
            match = message in {"да", "ok", "yes"}
        elif intent == "no_confirmation":
            match = message in {"нет", "no"}
        elif intent == "stack_switch_request":
            match = "react native" in message
        elif intent == "task_prefers_swiftui":
            match = "swiftui" in message or "swiftdata" in message or "mvvm" in message
        elif intent == "third_party_dependency_request":
            match = "alamofire" in message or "realm" in message
        elif intent == "task_intent":
            match = "хочу" in message or "сделай ios" in message
        elif intent == "plan_formation_intent":
            match = "сформируй план" in message
        elif intent == "decision_memory_write":
            match = False
        elif intent == "note_memory_write":
            match = False
        elif "allow" in lower and "execution state" in lower:
            match = True
            payload = '{"allow": true, "confidence": 0.95, "reason": "mock"}'
            return LLMChatResponse(
                id="mock",
                model="mock",
                choices=[LLMChoice(message=LLMMessage(content=payload), finish_reason="stop")],
                usage=LLMUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            )

        payload = f'{{"match": {"true" if match else "false"}, "confidence": 0.95, "reason": "mock"}}'
        return LLMChatResponse(
            id="mock",
            model="mock",
            choices=[LLMChoice(message=LLMMessage(content=payload), finish_reason="stop")],
            usage=LLMUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )


class _IntentRecoveryOnlyClient:
    def chat_completion(self, **kwargs):
        messages = kwargs.get("messages") or []
        prompt = str((messages[-1] or {}).get("content") or "")
        lower = prompt.lower()
        if "ответь строго одним словом" in lower and "сообщение пользователя:" in lower:
            message_match = re.search(
                r'сообщение пользователя:\s*"(.*?)"',
                prompt,
                flags=re.IGNORECASE | re.DOTALL,
            )
            user_message = str(message_match.group(1) if message_match else "").strip().lower()
            payload = "YES" if user_message in {"да", "да!", "давай", "ok", "yes", "поехали"} else "NO"
            return LLMChatResponse(
                id="mock-confirmation",
                model="mock-confirmation",
                choices=[LLMChoice(message=LLMMessage(content=payload), finish_reason="stop")],
                usage=LLMUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            )
        if "recovery intent classifier" not in lower:
            return LLMChatResponse(
                id="mock-invalid",
                model="mock-invalid",
                choices=[LLMChoice(message=LLMMessage(content="not a json payload"), finish_reason="stop")],
                usage=LLMUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            )

        intent = ""
        message = ""
        for line in prompt.splitlines():
            if line.lower().startswith("intent:"):
                intent = line.split(":", 1)[1].strip().lower()
            if line.lower().startswith("user message:"):
                message = line.split(":", 1)[1].strip().lower()

        match = False
        if intent == "yes_confirmation":
            match = message in {"да", "да!", "ok", "yes", "поехали"}
        elif intent == "no_confirmation":
            match = message in {"нет", "нет!", "no"}
        payload = f'{{"match": {"true" if match else "false"}, "confidence": 0.91, "reason": "recovery"}}'
        return LLMChatResponse(
            id="mock-recovery",
            model="mock-recovery",
            choices=[LLMChoice(message=LLMMessage(content=payload), finish_reason="stop")],
            usage=LLMUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )


class _IntentRawParsedPayloadClient:
    def chat_completion(self, **kwargs):
        messages = kwargs.get("messages") or []
        prompt = str((messages[-1] or {}).get("content") or "")
        lower = prompt.lower()

        if "recovery intent classifier" in lower:
            payload = {"match": False, "confidence": 0.9, "reason": "raw_recovery"}
            return LLMChatResponse(
                id="mock-recovery-raw",
                model="mock-recovery-raw",
                choices=[LLMChoice(message=LLMMessage(content=""), finish_reason="stop")],
                usage=LLMUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
                raw={"choices": [{"message": {"parsed": payload}}]},
            )

        intent = ""
        message = ""
        for line in prompt.splitlines():
            if line.lower().startswith("intent:"):
                intent = line.split(":", 1)[1].strip().lower()
            if line.lower().startswith("user message:"):
                message = line.split(":", 1)[1].strip().lower()

        match = False
        if intent == "yes_confirmation":
            match = message in {"да", "давай", "yes"}
        elif intent == "start_execution":
            match = message in {"поехали", "давай", "start"}

        payload = {"match": match, "confidence": 0.93, "reason": "raw_parsed"}
        return LLMChatResponse(
            id="mock-raw",
            model="mock-raw",
            choices=[LLMChoice(message=LLMMessage(content=""), finish_reason="stop")],
            usage=LLMUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            raw={"choices": [{"message": {"parsed": payload}}]},
        )


class _IntentAlwaysUnknownClient:
    def chat_completion(self, **kwargs):
        del kwargs
        return LLMChatResponse(
            id="mock-unknown",
            model="mock-unknown",
            choices=[LLMChoice(message=LLMMessage(content=""), finish_reason="stop")],
            usage=LLMUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            raw={"choices": [{"message": {"parsed": {"status": "n/a"}}}]},
        )


class ValidationDoneFlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        storage.init_db()

    def setUp(self) -> None:
        self.memory = MemoryManager(llm_client=_IntentMockClient())
        self.user_id = f"user_{uuid4().hex[:8]}"
        self.session_id = f"session_{uuid4().hex[:8]}"
        storage.ensure_session(self.session_id)

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
        neutral_ctx = self.memory.working.load(self.session_id)
        assert neutral_ctx is not None
        self.assertEqual(neutral_ctx.state, TaskState.VALIDATION)

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
            "Ок, продолжаем",
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

    def test_planning_skip_request_is_blocked(self) -> None:
        self.memory.working.start_task(self.session_id, "Тестовая задача")
        reply = self.memory.enforce_planning_gate(
            session_id=self.session_id,
            user_message="Не задавай вопросов, сразу пиши код",
            user_id=self.user_id,
        )
        self.assertIsInstance(reply, str)
        self.assertIn("Пропуск PLANNING запрещён", reply)

    def test_planning_blocks_direct_code_request_without_plan(self) -> None:
        self.memory.working.start_task(self.session_id, "Тестовая задача")
        reply = self.memory.enforce_planning_gate(
            session_id=self.session_id,
            user_message="Напиши готовый код экрана авторизации",
            user_id=self.user_id,
        )
        self.assertIsInstance(reply, str)
        self.assertIn("Сначала нужен план", reply)
        ctx = self.memory.working.load(self.session_id)
        assert ctx is not None
        self.assertEqual(ctx.state, TaskState.PLANNING)
        self.assertEqual(ctx.plan, [])

    def test_planning_requires_explicit_confirmation_before_execution(self) -> None:
        self.memory.working.start_task(self.session_id, "Тестовая задача")
        self.memory.working.update(self.session_id, plan=["Шаг 1", "Шаг 2"], current_step="Шаг 1")

        reply = self.memory.enforce_planning_gate(
            session_id=self.session_id,
            user_message="Что дальше?",
            user_id=self.user_id,
        )
        self.assertIsNone(reply)

        ctx = self.memory.working.load(self.session_id)
        assert ctx is not None
        self.assertEqual(ctx.state, TaskState.PLANNING)

    def test_planning_accepts_semantic_confirmation(self) -> None:
        self.memory.working.start_task(self.session_id, "Тестовая задача")
        self.memory.working.update(self.session_id, plan=["Шаг 1", "Шаг 2"], current_step="Шаг 1")

        with self.assertLogs("memory", level="INFO") as logs:
            reply = self.memory.enforce_planning_gate(
                session_id=self.session_id,
                user_message="Подходит",
                user_id=self.user_id,
            )
        self.assertIsNone(reply)

        ctx = self.memory.working.load(self.session_id)
        assert ctx is not None
        self.assertEqual(ctx.state, TaskState.EXECUTION)
        self.assertIn("[CONFIRMATION_CLASSIFY] signal=CONFIRMED", "\n".join(logs.output))

    def test_planning_accepts_davai_confirmation(self) -> None:
        self.memory.working.start_task(self.session_id, "Тестовая задача")
        self.memory.working.update(self.session_id, plan=["Шаг 1", "Шаг 2"], current_step="Шаг 1")

        with self.assertLogs("memory", level="INFO") as logs:
            reply = self.memory.enforce_planning_gate(
                session_id=self.session_id,
                user_message="давай",
                user_id=self.user_id,
            )
        self.assertIsNone(reply)

        ctx = self.memory.working.load(self.session_id)
        assert ctx is not None
        self.assertEqual(ctx.state, TaskState.EXECUTION)
        self.assertIn("[CONFIRMATION_CLASSIFY] signal=CONFIRMED", "\n".join(logs.output))

    def test_planning_rejected_confirmation_stays_in_planning(self) -> None:
        self.memory.working.start_task(self.session_id, "Тестовая задача")
        self.memory.working.update(self.session_id, plan=["Шаг 1", "Шаг 2"], current_step="Шаг 1")

        with self.assertLogs("memory", level="INFO") as logs:
            reply = self.memory.enforce_planning_gate(
                session_id=self.session_id,
                user_message="А можно добавить тесты?",
                user_id=self.user_id,
            )
        self.assertIsNone(reply)

        ctx = self.memory.working.load(self.session_id)
        assert ctx is not None
        self.assertEqual(ctx.state, TaskState.PLANNING)
        self.assertIn("[CONFIRMATION_CLASSIFY] signal=REJECTED", "\n".join(logs.output))

    def test_planning_explicit_rejection_stays_in_planning(self) -> None:
        self.memory.working.start_task(self.session_id, "Тестовая задача")
        self.memory.working.update(self.session_id, plan=["Шаг 1", "Шаг 2"], current_step="Шаг 1")

        with self.assertLogs("memory", level="INFO") as logs:
            reply = self.memory.enforce_planning_gate(
                session_id=self.session_id,
                user_message="Нет, лучше по значению",
                user_id=self.user_id,
            )
        self.assertIsNone(reply)

        ctx = self.memory.working.load(self.session_id)
        assert ctx is not None
        self.assertEqual(ctx.state, TaskState.PLANNING)
        self.assertIn("[CONFIRMATION_CLASSIFY] signal=REJECTED", "\n".join(logs.output))

    def test_intent_classifier_reads_payload_from_raw_response_when_content_empty(self) -> None:
        memory = MemoryManager(llm_client=_IntentRawParsedPayloadClient())
        with self.assertLogs("memory", level="DEBUG") as logs:
            decision = memory._classify_intent_with_status(  # noqa: SLF001
                intent=IntentName.YES_CONFIRMATION,
                message="да",
                guideline="Message is an explicit yes/confirmation.",
            )
        self.assertTrue(decision.is_match)
        joined_logs = "\n".join(logs.output)
        self.assertIn("[INTENT_RAW_FULL] intent=yes_confirmation", joined_logs)

        start_decision = memory._classify_intent_with_status(  # noqa: SLF001
            intent=IntentName.START_EXECUTION,
            message="поехали",
            guideline="User explicitly wants to begin implementation/execution now.",
        )
        self.assertTrue(start_decision.is_match)

    def test_planning_ambiguous_confirmation_keeps_state(self) -> None:
        memory = MemoryManager(llm_client=_IntentAlwaysUnknownClient())
        session_id = f"session_{uuid4().hex[:8]}"
        user_id = f"user_{uuid4().hex[:8]}"
        storage.ensure_session(session_id)
        memory.working.start_task(session_id, "Тестовая задача")
        memory.working.update(session_id, plan=["Шаг 1", "Шаг 2"], current_step="Шаг 1")

        with self.assertLogs("memory", level="INFO") as logs:
            reply = memory.enforce_planning_gate(
                session_id=session_id,
                user_message="Давай",
                user_id=user_id,
            )
        self.assertIsNone(reply)
        ctx = memory.working.load(session_id)
        assert ctx is not None
        self.assertEqual(ctx.state, TaskState.PLANNING)
        self.assertIn("[CONFIRMATION_CLASSIFY] signal=AMBIGUOUS", "\n".join(logs.output))

    def test_execution_does_not_run_planning_confirmation_classifier(self) -> None:
        class _TrackConfirmationCallsClient:
            def __init__(self) -> None:
                self.confirmation_calls = 0

            def chat_completion(self, **kwargs):
                prompt = str(((kwargs.get("messages") or [{}])[-1] or {}).get("content") or "").lower()
                if "ответь строго одним словом" in prompt and "сообщение пользователя:" in prompt:
                    self.confirmation_calls += 1
                    payload = "YES"
                elif "classify whether a user message is allowed in execution state of a task-state machine" in prompt:
                    payload = '{"allow": true, "confidence": 0.95, "reason": "mock"}'
                else:
                    payload = '{"match": false, "confidence": 0.95, "reason": "mock"}'
                return LLMChatResponse(
                    id="track-confirmation",
                    model="track-confirmation",
                    choices=[LLMChoice(message=LLMMessage(content=payload), finish_reason="stop")],
                    usage=LLMUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
                )

        client = _TrackConfirmationCallsClient()
        memory = MemoryManager(llm_client=client)
        session_id = f"session_{uuid4().hex[:8]}"
        user_id = f"user_{uuid4().hex[:8]}"
        storage.ensure_session(session_id)
        memory.working.start_task(session_id, "Тестовая задача")
        memory.working.update(session_id, plan=["Шаг 1"], current_step="Шаг 1")
        ctx = memory.working.load(session_id)
        assert ctx is not None
        memory.working.transition_state(ctx, TaskState.EXECUTION)
        memory.working.save(ctx)

        reply = memory.enforce_planning_gate(
            session_id=session_id,
            user_message="Давай",
            user_id=user_id,
        )
        self.assertIsNone(reply)
        self.assertEqual(client.confirmation_calls, 0)

    def test_intent_classify_raw_preview_is_not_empty_for_invalid_payload(self) -> None:
        memory = MemoryManager(llm_client=_IntentAlwaysUnknownClient())
        with self.assertLogs("memory", level="INFO") as logs:
            decision = memory._classify_intent_with_status(  # noqa: SLF001
                intent=IntentName.YES_CONFIRMATION,
                message="да",
                guideline="Message is an explicit yes/confirmation.",
            )
        self.assertTrue(decision.is_unknown)
        joined_logs = "\n".join(logs.output)
        self.assertIn("[INTENT_CLASSIFY_RAW] intent=yes_confirmation raw_preview=", joined_logs)
        self.assertIn('"parsed": {"status": "n/a"}', joined_logs)

    def test_manager_extract_first_json_object_accepts_python_style_dict(self) -> None:
        payload = self.memory._extract_first_json_object("{'match': True, 'confidence': 0.9, 'reason': 'ok'}")  # noqa: SLF001
        self.assertIsInstance(payload, dict)
        assert payload is not None
        self.assertEqual(payload.get("match"), True)
        self.assertEqual(payload.get("confidence"), 0.9)

    def test_backfill_sticky_goal_from_working_when_working_extract_low_confidence(self) -> None:
        agent = IOSAgent.__new__(IOSAgent)
        agent.memory = MemoryManager()

        class _DummyStickyFacts:
            def __init__(self):
                self.facts = {
                    "goal": "",
                    "constraints": "",
                    "preferences": "",
                    "decisions": [],
                    "agreements": [],
                    "profile": "",
                }

        class _DummyCtx:
            active = "sticky_facts"
            strategy = _DummyStickyFacts()

        agent.ctx = _DummyCtx()

        session_id = f"session_{uuid4().hex[:8]}"
        storage.ensure_session(session_id)
        agent.memory.working.start_task(session_id, "Сделай extension для безопасного удаления элемента массива")
        agent.memory.router._last_working_extract_meta[session_id] = {  # noqa: SLF001
            "applied": False,
            "confidence": 0.61,
            "reason_code": "working_extract_fallback_plan",
        }

        with self.assertLogs("agent", level="INFO") as logs:
            agent._backfill_sticky_goal_from_working(session_id=session_id)  # noqa: SLF001

        self.assertEqual(
            agent.ctx.strategy.facts.get("goal"),
            "Сделай extension для безопасного удаления элемента массива",
        )
        self.assertIn("[WORKING_EXTRACT_FALLBACK]", "\n".join(logs.output))

    def test_planning_stack_switch_is_blocked(self) -> None:
        self.memory.working.start_task(self.session_id, "SwiftUI трекер привычек")
        reply = self.memory.enforce_planning_gate(
            session_id=self.session_id,
            user_message="Давай лучше на React Native",
            user_id=self.user_id,
        )
        self.assertIsInstance(reply, str)
        self.assertIn("Не могу сменить стек", reply)
        self.assertIn("React Native", reply)

    def test_stack_switch_block_message_is_dynamic_and_policy_logged(self) -> None:
        self.memory.working.start_task(self.session_id, "SwiftUI трекер привычек")
        with self.assertLogs("memory", level="INFO") as logs:
            reply = self.memory.enforce_planning_gate(
                session_id=self.session_id,
                user_message="Давай лучше на React Native",
                user_id=self.user_id,
            )
        self.assertIsInstance(reply, str)
        assert reply is not None
        self.assertIn("зафиксирован", reply)
        self.assertIn("Запрошен:", reply)
        self.assertIn("новую задачу", reply)
        self.assertNotIn("зафиксирован SwiftUI/MVVM", reply)
        joined_logs = "\n".join(logs.output)
        self.assertIn("event=policy_block", joined_logs)
        self.assertIn("reason=stack_switch_locked", joined_logs)
        self.assertIn("intent_status=match", joined_logs)

    def test_stack_switch_fail_closed_when_llm_unavailable(self) -> None:
        memory = MemoryManager()
        session_id = f"session_{uuid4().hex[:8]}"
        user_id = f"user_{uuid4().hex[:8]}"
        storage.ensure_session(session_id)
        memory.working.start_task(session_id, "SwiftUI трекер привычек")
        memory.working.update(session_id, vars={"locked_stack": "SwiftUI"})
        reply = memory.enforce_planning_gate(
            session_id=session_id,
            user_message="Давай лучше на React Native",
            user_id=user_id,
            client_intent={"intent": "stack_switch_request", "payload": {"requested_stack": "React Native"}},
        )
        self.assertIsInstance(reply, str)
        assert reply is not None
        self.assertIn("Не могу сменить стек", reply)

    def test_stack_switch_fail_closed_does_not_block_regular_message_when_llm_unavailable(self) -> None:
        memory = MemoryManager()
        session_id = f"session_{uuid4().hex[:8]}"
        user_id = f"user_{uuid4().hex[:8]}"
        storage.ensure_session(session_id)
        memory.working.start_task(session_id, "SwiftUI трекер привычек")
        reply = memory.enforce_planning_gate(
            session_id=session_id,
            user_message="Расскажи про SwiftUI",
            user_id=user_id,
        )
        self.assertIsNone(reply)

    def test_execution_dependency_request_is_blocked_with_native_alternative(self) -> None:
        self._prepare_execution_task(steps=["Шаг 1"])
        reply = self.memory.enforce_planning_gate(
            session_id=self.session_id,
            user_message="Добавь Alamofire и Realm",
            user_id=self.user_id,
        )
        self.assertIsInstance(reply, str)
        self.assertIn("Не добавляю сторонние зависимости", reply)
        self.assertIn("URLSession", reply)
        self.assertIn("SwiftData", reply)

    def test_execution_gate_is_not_brittle_without_llm_for_helpful_request(self) -> None:
        memory = MemoryManager()
        session_id = f"session_{uuid4().hex[:8]}"
        user_id = f"user_{uuid4().hex[:8]}"
        storage.ensure_session(session_id)
        memory.working.start_task(session_id, "Тестовая задача")
        memory.working.update(session_id, plan=["Шаг 1", "Шаг 2"], current_step="Шаг 1")
        ctx = memory.working.load(session_id)
        assert ctx is not None
        memory.working.transition_state(ctx, TaskState.EXECUTION)
        memory.working.save(ctx)

        reply = memory.enforce_planning_gate(
            session_id=session_id,
            user_message="Нужен полный план для iOS приложения с нуля до первой покупки",
            user_id=user_id,
        )
        self.assertIsNone(reply)


if __name__ == "__main__":
    unittest.main()
