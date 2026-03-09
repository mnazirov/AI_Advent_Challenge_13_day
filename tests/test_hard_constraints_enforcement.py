from __future__ import annotations

import unittest

from agent import IOSAgent
from memory.prompt_builder import PromptBuilder


class _StubMemory:
    def __init__(self, constraints: list[str], *, verified: bool = True) -> None:
        self.constraints = list(constraints)
        self.verified = bool(verified)

    def get_profile_snapshot(self, *, user_id: str, session_id: str | None = None) -> dict:
        _ = user_id, session_id
        return {
            "hard_constraints": {
                "value": list(self.constraints),
                "verified": self.verified,
            }
        }


class HardConstraintsEnforcementTests(unittest.TestCase):
    def test_prompt_injects_hard_constraints_before_base_behavior(self) -> None:
        builder = PromptBuilder()
        messages, _preview = builder.build(
            system_instructions="You are an iOS assistant.",
            data_context="",
            long_term={
                "profile": {
                    "hard_constraints": {
                        "value": ["Только SwiftUI", "iOS 16", "Без сторонних зависимостей"],
                        "source": "user_explicit",
                        "verified": True,
                        "confidence": None,
                        "updated_at": "2026-03-05T00:00:00",
                    }
                }
            },
            working=None,
            short_term_messages=[],
            user_query="Сделай экран логина",
        )
        system_prompt = str(messages[0]["content"])
        self.assertIn("[HARD_CONSTRAINTS]", system_prompt)
        self.assertIn("[BASE_BEHAVIOR]", system_prompt)
        self.assertLess(system_prompt.index("[HARD_CONSTRAINTS]"), system_prompt.index("[BASE_BEHAVIOR]"))
        self.assertIn("Never violate them", system_prompt)
        self.assertIn("Name the specific constraint", system_prompt)
        self.assertIn("Propose a compliant alternative", system_prompt)
        self.assertIn("[ROLE]", system_prompt)
        self.assertIn("[PROFILE]", system_prompt)
        self.assertIn("[STATE]", system_prompt)
        self.assertIn("[INVARIANTS]", system_prompt)
        self.assertIn("[RULES]", system_prompt)
        self.assertIn("[USER QUERY]", system_prompt)
        self.assertIn("[STEP_DONE: N]", system_prompt)
        self.assertIn("[NEXT_STATE: PLANNING]", system_prompt)
        self.assertIn("[NEXT_STATE: EXECUTION]", system_prompt)
        self.assertIn("[NEXT_STATE: VALIDATION]", system_prompt)
        self.assertIn("[NEXT_STATE: DONE]", system_prompt)
        self.assertIn("[VALIDATION_OK]", system_prompt)
        self.assertIn("[VALIDATION_FAIL: причина]", system_prompt)
        self.assertIn("[DONE: описание]", system_prompt)
        self.assertIn("[OPEN_QUESTION: текст]", system_prompt)
        self.assertIn("[CODE_ARTIFACT]", system_prompt)
        self.assertIn("КРИТИЧНО — ДОПУСТИМЫЕ МАРКЕРЫ:", system_prompt)
        self.assertIn("В PLANNING формируй краткий согласованный план шагов", system_prompt)
        self.assertIn("Переход в EXECUTION допустим только после явного подтверждения плана", system_prompt)
        self.assertIn("use [NEXT_STATE: PLANNING] ONLY if user explicitly asks", system_prompt)

    def test_agent_refuses_and_names_violated_constraint(self) -> None:
        agent = IOSAgent.__new__(IOSAgent)
        agent.memory = _StubMemory(["Только SwiftUI"])
        reply = agent._enforce_hard_constraints_on_reply(
            user_id="u1",
            reply="Используем UIKit:\nimport UIKit\nclass LoginVC: UIViewController {}",
        )
        self.assertIn("«Только SwiftUI»", reply)
        self.assertIn("Почему:", reply)
        self.assertIn("Вместо этого предлагаю:", reply)
        self.assertIn("SwiftUI", reply)

    def test_agent_ignores_unverified_hard_constraints(self) -> None:
        agent = IOSAgent.__new__(IOSAgent)
        agent.memory = _StubMemory(["Только SwiftUI"], verified=False)
        reply = agent._enforce_hard_constraints_on_reply(
            user_id="u1",
            reply="Используем UIKit:\nimport UIKit\nclass LoginVC: UIViewController {}",
        )
        self.assertIn("UIKit", reply)
        self.assertNotIn("Не могу выполнить запрос", reply)

    def test_mvvm_constraint_does_not_treat_product_mvp_as_architecture(self) -> None:
        agent = IOSAgent.__new__(IOSAgent)
        violation = agent._detect_constraint_violation(
            constraint="Архитектура MVVM",
            reply="MVP согласован, давай зафиксируем scope и экраны.",
        )
        self.assertIsNone(violation)

if __name__ == "__main__":
    unittest.main()
