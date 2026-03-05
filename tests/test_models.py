from __future__ import annotations

import unittest

from agent import IOSAgent
from llm.client import LLMChatResponse, LLMChoice, LLMMessage, LLMUsage


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


if __name__ == "__main__":
    unittest.main()
