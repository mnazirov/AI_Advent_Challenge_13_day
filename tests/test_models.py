from __future__ import annotations

import unittest
from types import SimpleNamespace
from uuid import uuid4

import storage
from agent import IOSAgent
from context_strategies import ContextStrategyManager
from llm.client import LLMChatResponse, LLMChoice, LLMMessage, LLMUsage
from llm.openai_client import OpenAILLMClient
from memory.manager import MemoryManager
from memory.models import TaskState


class _FakeModelNotFoundError(Exception):
    def __init__(self, message: str):
        super().__init__(message)
        self.code = "model_not_found"


class _StubCtx:
    def __init__(self) -> None:
        self.last_model: str | None = None

    def set_model(self, model: str) -> None:
        self.last_model = model


class _FailThenSuccessClient:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def chat_completion(self, **kwargs):
        self.calls.append(dict(kwargs))
        if len(self.calls) == 1:
            raise _FakeModelNotFoundError("The model does not exist or you do not have access to it.")
        return LLMChatResponse(
            id="ok",
            model=str(kwargs.get("model") or ""),
            choices=[LLMChoice(message=LLMMessage(content="ok"))],
            usage=LLMUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )


class _PlanningFlowLLMClient:
    def __init__(self) -> None:
        self.main_calls = 0

    def chat_completion(self, **kwargs):
        messages = kwargs.get("messages") or []
        prompt = str((messages[-1] or {}).get("content") or "")
        lower = prompt.lower()
        max_tokens = int(kwargs.get("max_tokens") or kwargs.get("max_completion_tokens") or 0)

        if max_tokens == 4096:
            self.main_calls += 1
            payload = (
                "<internal>\n"
                "[STEP_DONE: 0]\n"
                "[NEXT_STATE: PLANNING]\n"
                "[VALIDATION_OK]\n"
                "</internal>\n"
                "<external>\n"
                "Начнём с уточнения требований к extension и edge-case поведения.\n"
                "</external>"
            )
            return LLMChatResponse(
                id="mock-main",
                model="mock-main",
                choices=[LLMChoice(message=LLMMessage(content=payload), finish_reason="stop")],
                usage=LLMUsage(prompt_tokens=20, completion_tokens=20, total_tokens=40),
            )

        if "information extraction system for ios development conversations" in lower:
            payload = (
                '{"goal":"","constraints":"","preferences":"","decisions":[],"agreements":[],"profile":""}'
            )
            return LLMChatResponse(
                id="mock-facts",
                model="mock-facts",
                choices=[LLMChoice(message=LLMMessage(content=payload), finish_reason="stop")],
                usage=LLMUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            )

        if "extract working-memory task updates from a single user message." in lower:
            payload = (
                '{"is_working_update": false, "task": "", "plan": [], "plan_steps_to_add": [], '
                '"current_step": "", "done_steps_to_add": [], "requirements_to_add": [], '
                '"artifacts_to_add": [], "confidence": 0.61}'
            )
            return LLMChatResponse(
                id="mock-working",
                model="mock-working",
                choices=[LLMChoice(message=LLMMessage(content=payload), finish_reason="stop")],
                usage=LLMUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            )

        if "you are an intent classifier for a task state machine." in lower:
            payload = '{"match": false, "confidence": 0.95, "reason": "mock"}'
            return LLMChatResponse(
                id="mock-intent",
                model="mock-intent",
                choices=[LLMChoice(message=LLMMessage(content=payload), finish_reason="stop")],
                usage=LLMUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            )

        if "classify whether a user message is allowed in execution state of a task-state machine" in lower:
            payload = '{"allow": true, "confidence": 0.95, "reason": "mock"}'
            return LLMChatResponse(
                id="mock-policy",
                model="mock-policy",
                choices=[LLMChoice(message=LLMMessage(content=payload), finish_reason="stop")],
                usage=LLMUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            )

        payload = '{"match": false, "confidence": 0.95, "reason": "mock"}'
        return LLMChatResponse(
            id="mock-default",
            model="mock-default",
            choices=[LLMChoice(message=LLMMessage(content=payload), finish_reason="stop")],
            usage=LLMUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )


class _PlanningInvariantRecoveryLLMClient:
    def __init__(self) -> None:
        self.main_calls = 0

    def chat_completion(self, **kwargs):
        messages = kwargs.get("messages") or []
        prompt = str((messages[-1] or {}).get("content") or "")
        lower = prompt.lower()
        max_tokens = int(kwargs.get("max_tokens") or kwargs.get("max_completion_tokens") or 0)

        if max_tokens == 4096:
            self.main_calls += 1
            if self.main_calls <= 2:
                payload = (
                    "<internal>\n"
                    "[STEP_DONE: 0]\n"
                    "[NEXT_STATE: PLANNING]\n"
                    "[VALIDATION_OK]\n"
                    "</internal>\n"
                    "<external>\n"
                    "```swift\n"
                    "extension Array {\n"
                    "    mutating func safeRemove(at index: Int) {\n"
                    "        guard indices.contains(index) else { return }\n"
                    "        remove(at: index)\n"
                    "    }\n"
                    "}\n"
                    "```\n"
                    "</external>"
                )
            else:
                payload = (
                    "<internal>\n"
                    "[STEP_DONE: 0]\n"
                    "[NEXT_STATE: EXECUTION]\n"
                    "[VALIDATION_OK]\n"
                    "</internal>\n"
                    "<external>\n"
                    "Продолжаем выполнение шага в EXECUTION.\n"
                    "</external>"
                )
            return LLMChatResponse(
                id=f"mock-main-{self.main_calls}",
                model="mock-main",
                choices=[LLMChoice(message=LLMMessage(content=payload), finish_reason="stop")],
                usage=LLMUsage(prompt_tokens=20, completion_tokens=20, total_tokens=40),
            )

        if "ответь строго одним словом" in lower and "сообщение пользователя:" in lower:
            return LLMChatResponse(
                id="mock-confirmation",
                model="mock-confirmation",
                choices=[LLMChoice(message=LLMMessage(content="NO"), finish_reason="stop")],
                usage=LLMUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            )

        if "information extraction system for ios development conversations" in lower:
            payload = (
                '{"goal":"","constraints":"","preferences":"","decisions":[],"agreements":[],"profile":""}'
            )
            return LLMChatResponse(
                id="mock-facts",
                model="mock-facts",
                choices=[LLMChoice(message=LLMMessage(content=payload), finish_reason="stop")],
                usage=LLMUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            )

        if "extract working-memory task updates from a single user message." in lower:
            payload = (
                '{"is_working_update": false, "task": "", "plan": [], "plan_steps_to_add": [], '
                '"current_step": "", "done_steps_to_add": [], "requirements_to_add": [], '
                '"artifacts_to_add": [], "confidence": 0.61}'
            )
            return LLMChatResponse(
                id="mock-working",
                model="mock-working",
                choices=[LLMChoice(message=LLMMessage(content=payload), finish_reason="stop")],
                usage=LLMUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            )

        if "you are an intent classifier for a task state machine." in lower:
            payload = '{"match": false, "confidence": 0.95, "reason": "mock"}'
            return LLMChatResponse(
                id="mock-intent",
                model="mock-intent",
                choices=[LLMChoice(message=LLMMessage(content=payload), finish_reason="stop")],
                usage=LLMUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            )

        if "classify whether a user message is allowed in execution state of a task-state machine" in lower:
            payload = '{"allow": true, "confidence": 0.95, "reason": "mock"}'
            return LLMChatResponse(
                id="mock-policy",
                model="mock-policy",
                choices=[LLMChoice(message=LLMMessage(content=payload), finish_reason="stop")],
                usage=LLMUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            )

        payload = '{"match": false, "confidence": 0.95, "reason": "mock"}'
        return LLMChatResponse(
            id="mock-default",
            model="mock-default",
            choices=[LLMChoice(message=LLMMessage(content=payload), finish_reason="stop")],
            usage=LLMUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )


class _RollbackValidationLLMClient:
    def chat_completion(self, **kwargs):
        prompt = str(((kwargs.get("messages") or [{}])[-1] or {}).get("content") or "")
        lower = prompt.lower()
        if "пользователь просит пересмотреть весь план работы?" in lower:
            if "переделаем план" in lower or "начнём заново" in lower:
                payload = "YES"
            else:
                payload = "NO"
            return LLMChatResponse(
                id="mock-rollback-validate",
                model="mock-rollback-validate",
                choices=[LLMChoice(message=LLMMessage(content=payload), finish_reason="stop")],
                usage=LLMUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            )
        return LLMChatResponse(
            id="mock-default",
            model="mock-default",
            choices=[LLMChoice(message=LLMMessage(content="NO"), finish_reason="stop")],
            usage=LLMUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )


class ModelRegistryTests(unittest.TestCase):
    def test_default_model_is_gpt_5_3_instant(self) -> None:
        self.assertEqual(IOSAgent.DEFAULT_MODEL, "gpt-5.3-instant")

    def test_gpt_5_3_instant_is_available(self) -> None:
        models = IOSAgent.available_models()
        self.assertIn("gpt-5.3-instant", models)

    def test_validate_accepts_gpt_5_3_instant(self) -> None:
        validated = IOSAgent._validate_model("gpt-5.3-instant")
        self.assertEqual(validated, "gpt-5.3-instant")

    def test_model_not_found_falls_back_to_gpt_5_mini(self) -> None:
        agent = IOSAgent.__new__(IOSAgent)
        agent.llm_client = _FailThenSuccessClient()
        agent.ctx = _StubCtx()
        agent.model = "gpt-5.3-instant"

        response = agent._create_chat_completion(
            model="gpt-5.3-instant",
            messages=[{"role": "user", "content": "ping"}],
        )

        self.assertEqual(agent.model, "gpt-5-mini")
        self.assertEqual(agent.ctx.last_model, "gpt-5-mini")
        self.assertEqual(len(agent.llm_client.calls), 2)
        self.assertEqual(agent.llm_client.calls[1].get("model"), "gpt-5-mini")
        self.assertEqual(response.model, "gpt-5-mini")


class OpenAIClientContentNormalizationTests(unittest.TestCase):
    def test_convert_response_flattens_segmented_text_parts(self) -> None:
        client = OpenAILLMClient.__new__(OpenAILLMClient)
        raw = SimpleNamespace(
            id="raw-id",
            model="gpt-5.3-instant",
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(
                        content=[
                            {"type": "text", "text": '{"match": true, '},
                            {"type": "text", "text": '"confidence": 0.91, "reason": "ok"}'},
                        ]
                    ),
                )
            ],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=2, total_tokens=3),
        )

        converted = client._convert_response(raw)
        content = converted.choices[0].message.content

        self.assertEqual(content, '{"match": true, "confidence": 0.91, "reason": "ok"}')

    def test_convert_response_reads_output_when_message_content_is_empty(self) -> None:
        client = OpenAILLMClient.__new__(OpenAILLMClient)
        raw = SimpleNamespace(
            id="raw-id",
            model="gpt-5.3-instant",
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(content=None),
                    output=[{"type": "text", "text": '{"match": true, "confidence": 0.95, "reason": "ok"}'}],
                )
            ],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=2, total_tokens=3),
        )

        converted = client._convert_response(raw)
        content = converted.choices[0].message.content

        self.assertEqual(content, '{"match": true, "confidence": 0.95, "reason": "ok"}')


class AgentRecallBehaviorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        storage.init_db()

    def test_restore_memory_session_hydrates_short_term(self) -> None:
        session_id = f"session_{uuid4().hex[:8]}"
        storage.ensure_session(session_id)
        agent = IOSAgent.__new__(IOSAgent)
        agent.memory = MemoryManager()
        agent.restore_memory_session(
            session_id=session_id,
            messages=[
                {"role": "user", "content": "Первый вопрос"},
                {"role": "assistant", "content": "Первый ответ"},
                {"role": "user", "content": "Второй вопрос"},
            ],
        )
        short_term = agent.memory.short_term.get_context(session_id)
        self.assertEqual([msg["content"] for msg in short_term], ["Первый вопрос", "Первый ответ", "Второй вопрос"])

    def test_memory_recall_response_summarizes_recent_user_turns(self) -> None:
        agent = IOSAgent.__new__(IOSAgent)
        agent.conversation_history = [
            {"role": "user", "content": "Как сделать onboarding?"},
            {"role": "assistant", "content": "Нужен flow из 3 экранов."},
            {"role": "user", "content": "Как подключить подписку?"},
            {"role": "assistant", "content": "Через StoreKit 2."},
        ]
        self.assertTrue(agent._is_memory_recall_request("Что я у тебя спрашивал?"))
        reply = agent._build_memory_recall_response()
        self.assertIn("Как сделать onboarding?", reply)
        self.assertIn("Как подключить подписку?", reply)

    def test_finalize_external_message_does_not_fail_on_unknown_next_state(self) -> None:
        session_id = f"session_{uuid4().hex[:8]}"
        user_id = f"user_{uuid4().hex[:8]}"
        storage.ensure_session(session_id)

        agent = IOSAgent.__new__(IOSAgent)
        agent.memory = MemoryManager()

        text, meta = agent._finalize_external_message(
            session_id=session_id,
            user_id=user_id,
            text="ok",
            internal_trace="",
            raw_response=(
                "<internal>[STEP_DONE: 0][NEXT_STATE: COLLECT_REQUIREMENTS][VALIDATION_OK]</internal>"
                "<external>Принято. Идём дальше.</external>"
            ),
            source="llm",
        )

        self.assertIn("Принято. Идём дальше.", text)
        self.assertNotIn("Не удалось безопасно сформировать", text)
        invariant_report = meta.get("invariant_report") or {}
        self.assertEqual(invariant_report.get("overall_status"), "ok")

    def test_finalize_external_message_normalizes_missing_markers_for_llm(self) -> None:
        session_id = f"session_{uuid4().hex[:8]}"
        user_id = f"user_{uuid4().hex[:8]}"
        storage.ensure_session(session_id)

        agent = IOSAgent.__new__(IOSAgent)
        agent.memory = MemoryManager()

        text, meta = agent._finalize_external_message(
            session_id=session_id,
            user_id=user_id,
            text="ok",
            internal_trace="",
            raw_response="<external>Нормальный пользовательский ответ без служебных маркеров.</external>",
            source="llm",
        )

        self.assertIn("Нормальный пользовательский ответ", text)
        self.assertNotIn("Не удалось безопасно сформировать", text)
        invariant_report = meta.get("invariant_report") or {}
        self.assertEqual(invariant_report.get("overall_status"), "ok")
        self.assertIn("missing_step_done", invariant_report.get("normalization") or [])

    def test_chat_first_message_uses_llm_in_planning_without_static_block_reply(self) -> None:
        session_id = f"session_{uuid4().hex[:8]}"
        user_id = f"user_{uuid4().hex[:8]}"
        storage.ensure_session(session_id)

        mock = _PlanningFlowLLMClient()
        agent = IOSAgent.__new__(IOSAgent)
        agent.llm_client = mock
        agent.model = IOSAgent.DEFAULT_MODEL
        agent.ctx = ContextStrategyManager(client=mock, model=agent.model)
        agent.memory = MemoryManager(
            short_term_limit=30,
            llm_client=mock,
            step_parser_model="gpt-5-nano",
        )
        agent.conversation_history = []
        agent.last_token_stats = None
        agent.last_memory_stats = None
        agent.last_prompt_preview = None
        agent.last_chat_response_meta = None

        reply = agent.chat(
            "Сделай extension для безопасного удаления элемента массива",
            session_id=session_id,
            user_id=user_id,
        )

        self.assertGreaterEqual(mock.main_calls, 1)
        self.assertNotIn("План сформирован. Подтвердите переход в EXECUTION", reply)
        self.assertIn("уточнения требований", reply.lower())
        self.assertNotEqual((agent.last_token_stats or {}).get("finish_reason"), "state_blocked_planning")

    def test_chat_invariant_fail_in_planning_recovers_via_execution_transition(self) -> None:
        session_id = f"session_{uuid4().hex[:8]}"
        user_id = f"user_{uuid4().hex[:8]}"
        storage.ensure_session(session_id)

        mock = _PlanningInvariantRecoveryLLMClient()
        agent = IOSAgent.__new__(IOSAgent)
        agent.llm_client = mock
        agent.model = IOSAgent.DEFAULT_MODEL
        agent.ctx = ContextStrategyManager(client=mock, model=agent.model)
        agent.memory = MemoryManager(
            short_term_limit=30,
            llm_client=mock,
            step_parser_model="gpt-5-nano",
        )
        agent.conversation_history = []
        agent.last_token_stats = None
        agent.last_memory_stats = None
        agent.last_prompt_preview = None
        agent.last_chat_response_meta = None

        agent.memory.working.start_task(session_id, "Тестовая задача")
        agent.memory.working.update(session_id, plan=["Шаг 1"], current_step="Шаг 1")

        reply = agent.chat(
            "Что дальше?",
            session_id=session_id,
            user_id=user_id,
        )

        self.assertIn("продолжаем выполнение шага", reply.lower())
        self.assertGreaterEqual(mock.main_calls, 3)
        self.assertNotEqual((agent.last_token_stats or {}).get("finish_reason"), "invariant_fail")
        ctx = agent.memory.working.load(session_id)
        assert ctx is not None
        self.assertEqual(ctx.state, TaskState.EXECUTION)

    def test_finalize_external_message_applies_step_done_to_working(self) -> None:
        session_id = f"session_{uuid4().hex[:8]}"
        user_id = f"user_{uuid4().hex[:8]}"
        storage.ensure_session(session_id)

        agent = IOSAgent.__new__(IOSAgent)
        agent.memory = MemoryManager()

        agent.memory.working.start_task(session_id, "Тестовая задача")
        agent.memory.working.update(session_id, plan=["Шаг 1", "Шаг 2", "Шаг 3"], current_step="Шаг 1")
        ctx = agent.memory.working.load(session_id)
        assert ctx is not None
        agent.memory.working.transition_state(ctx, TaskState.EXECUTION)
        agent.memory.working.save(ctx)

        with self.assertLogs("agent", level="INFO") as logs:
            text, meta = agent._finalize_external_message(
                session_id=session_id,
                user_id=user_id,
                text="ok",
                internal_trace="",
                raw_response=(
                    "<internal>[STEP_DONE: 1][NEXT_STATE: EXECUTION][VALIDATION_OK]</internal>"
                    "<external>Шаг зафиксирован.</external>"
                ),
                source="llm",
            )

        self.assertIn("Шаг зафиксирован.", text)
        self.assertEqual((meta.get("invariant_report") or {}).get("overall_status"), "ok")
        updated = agent.memory.working.load(session_id)
        assert updated is not None
        self.assertEqual(updated.done, ["Шаг 1"])
        self.assertEqual(updated.current_step, "Шаг 2")
        joined = "\n".join(logs.output)
        self.assertIn("[STEP_UPDATE] current_step=1", joined)
        self.assertIn("[STEP_WRITTEN] current_step=1", joined)

    def test_finalize_external_message_moves_execution_to_validation_on_last_step(self) -> None:
        session_id = f"session_{uuid4().hex[:8]}"
        user_id = f"user_{uuid4().hex[:8]}"
        storage.ensure_session(session_id)

        agent = IOSAgent.__new__(IOSAgent)
        agent.memory = MemoryManager()

        agent.memory.working.start_task(session_id, "Тестовая задача")
        agent.memory.working.update(session_id, plan=["Шаг 1", "Шаг 2", "Шаг 3"], current_step="Шаг 1")
        ctx = agent.memory.working.load(session_id)
        assert ctx is not None
        agent.memory.working.transition_state(ctx, TaskState.EXECUTION)
        agent.memory.working.save(ctx)
        agent.memory.working.complete_current_step(session_id)
        agent.memory.working.complete_current_step(session_id)

        with self.assertLogs("agent", level="INFO") as logs:
            text, meta = agent._finalize_external_message(
                session_id=session_id,
                user_id=user_id,
                text="ok",
                internal_trace="",
                raw_response=(
                    "<internal>[STEP_DONE: 3][NEXT_STATE: VALIDATION][VALIDATION_OK]</internal>"
                    "<external>Финальный шаг закрыт.</external>"
                ),
                source="llm",
            )

        self.assertIn("Финальный шаг закрыт.", text)
        self.assertEqual((meta.get("invariant_report") or {}).get("overall_status"), "ok")
        updated = agent.memory.working.load(session_id)
        assert updated is not None
        self.assertEqual(updated.state, TaskState.VALIDATION)
        joined = "\n".join(logs.output)
        self.assertIn("[STATE_UPDATE] EXECUTION -> VALIDATION", joined)

    def test_finalize_external_message_rolls_back_only_by_explicit_planning_marker(self) -> None:
        session_id = f"session_{uuid4().hex[:8]}"
        user_id = f"user_{uuid4().hex[:8]}"
        storage.ensure_session(session_id)

        agent = IOSAgent.__new__(IOSAgent)
        agent.memory = MemoryManager()
        agent.llm_client = _RollbackValidationLLMClient()
        agent.model = IOSAgent.DEFAULT_MODEL

        agent.memory.working.start_task(session_id, "Тестовая задача")
        agent.memory.working.update(session_id, plan=["Шаг 1", "Шаг 2"], current_step="Шаг 1")
        ctx = agent.memory.working.load(session_id)
        assert ctx is not None
        agent.memory.working.transition_state(ctx, TaskState.EXECUTION)
        agent.memory.working.save(ctx)

        # Question in EXECUTION should not trigger rollback by itself.
        text, meta = agent._finalize_external_message(
            session_id=session_id,
            user_id=user_id,
            text="ok",
            internal_trace="",
            raw_response=(
                "<internal>[STEP_DONE: 0][NEXT_STATE: EXECUTION][VALIDATION_OK]</internal>"
                "<external>В каком именно языке делаем extension?</external>"
            ),
            source="llm",
            last_user_message="какой именно язык?",
        )
        self.assertIn("языке", text.lower())
        self.assertEqual((meta.get("invariant_report") or {}).get("overall_status"), "ok")
        mid_ctx = agent.memory.working.load(session_id)
        assert mid_ctx is not None
        self.assertEqual(mid_ctx.state, TaskState.EXECUTION)

        # Explicit PLANNING marker should be suppressed when user did not request plan change.
        with self.assertLogs("agent", level="INFO") as logs_suppressed:
            text2, meta2 = agent._finalize_external_message(
                session_id=session_id,
                user_id=user_id,
                text="ok",
                internal_trace="",
                raw_response=(
                    "<internal>[STEP_DONE: 0][NEXT_STATE: PLANNING][VALIDATION_OK]</internal>"
                    "<external>Откатываемся к перепланированию.</external>"
                ),
                source="llm",
                last_user_message="Не, не нужно",
            )
        self.assertIn("перепланированию", text2.lower())
        self.assertEqual((meta2.get("invariant_report") or {}).get("overall_status"), "ok")
        suppressed_ctx = agent.memory.working.load(session_id)
        assert suppressed_ctx is not None
        self.assertEqual(suppressed_ctx.state, TaskState.EXECUTION)
        self.assertIn("[ROLLBACK_SUPPRESSED]", "\n".join(logs_suppressed.output))

        # Explicit marker should rollback only when user explicitly asks to replan.
        with self.assertLogs("agent", level="INFO") as logs_rollback:
            _text3, _meta3 = agent._finalize_external_message(
                session_id=session_id,
                user_id=user_id,
                text="ok",
                internal_trace="",
                raw_response=(
                    "<internal>[STEP_DONE: 0][NEXT_STATE: PLANNING][VALIDATION_OK]</internal>"
                    "<external>Перестраиваем план.</external>"
                ),
                source="llm",
                last_user_message="Давай переделаем план целиком",
            )
        final_ctx = agent.memory.working.load(session_id)
        assert final_ctx is not None
        self.assertEqual(final_ctx.state, TaskState.PLANNING)
        self.assertIn("[ROLLBACK_DECISION]", "\n".join(logs_rollback.output))

    def test_execution_guard_moves_to_validation_without_next_state_marker(self) -> None:
        session_id = f"session_{uuid4().hex[:8]}"
        user_id = f"user_{uuid4().hex[:8]}"
        storage.ensure_session(session_id)

        agent = IOSAgent.__new__(IOSAgent)
        agent.memory = MemoryManager()

        agent.memory.working.start_task(session_id, "Тестовая задача")
        agent.memory.working.update(session_id, plan=["Шаг 1"], current_step="Шаг 1")
        ctx = agent.memory.working.load(session_id)
        assert ctx is not None
        agent.memory.working.transition_state(ctx, TaskState.EXECUTION)
        agent.memory.working.save(ctx)

        with self.assertLogs("agent", level="INFO") as logs:
            text, meta = agent._finalize_external_message(
                session_id=session_id,
                user_id=user_id,
                text="ok",
                internal_trace="",
                raw_response=(
                    "<internal>[STEP_DONE: 1][VALIDATION_OK]</internal>"
                    "<external>Шаг завершён.</external>"
                ),
                source="llm",
            )

        self.assertIn("Шаг завершён.", text)
        self.assertEqual((meta.get("invariant_report") or {}).get("overall_status"), "ok")
        updated = agent.memory.working.load(session_id)
        assert updated is not None
        self.assertEqual(updated.state, TaskState.VALIDATION)
        joined = "\n".join(logs.output)
        self.assertIn("[EXECUTION_GUARD] завершено по счётчику: 1/1", joined)

    def test_execution_guard_moves_to_validation_by_semantic_done_signals(self) -> None:
        session_id = f"session_{uuid4().hex[:8]}"
        user_id = f"user_{uuid4().hex[:8]}"
        storage.ensure_session(session_id)

        agent = IOSAgent.__new__(IOSAgent)
        agent.memory = MemoryManager()
        agent.llm_client = _RollbackValidationLLMClient()
        agent.model = IOSAgent.DEFAULT_MODEL

        agent.memory.working.start_task(session_id, "Тестовая задача")
        agent.memory.working.update(session_id, plan=["Шаг 1", "Шаг 2", "Шаг 3"], current_step="Шаг 1")
        ctx = agent.memory.working.load(session_id)
        assert ctx is not None
        agent.memory.working.transition_state(ctx, TaskState.EXECUTION)
        agent.memory.working.save(ctx)
        agent.memory.working.complete_current_step(session_id)

        with self.assertLogs("agent", level="INFO") as logs:
            text, meta = agent._finalize_external_message(
                session_id=session_id,
                user_id=user_id,
                text="ok",
                internal_trace="",
                raw_response=(
                    "<internal>[STEP_DONE: 1][NEXT_STATE: EXECUTION][VALIDATION_OK]"
                    "[DONE: финальный код готов][CODE_ARTIFACT]</internal>"
                    "<external>```swift\nextension Array { }\n```\nГотово.</external>"
                ),
                source="llm",
                last_user_message="Не, не нужно",
            )

        self.assertIn("extension array", text.lower())
        self.assertEqual((meta.get("invariant_report") or {}).get("overall_status"), "ok")
        updated = agent.memory.working.load(session_id)
        assert updated is not None
        self.assertEqual(updated.state, TaskState.VALIDATION)
        self.assertEqual(len(updated.done), len(updated.plan))
        joined = "\n".join(logs.output)
        self.assertIn("[EXECUTION_GUARD] завершено семантически", joined)
        self.assertIn("[STATE_UPDATE] EXECUTION -> VALIDATION", joined)

if __name__ == "__main__":
    unittest.main()
