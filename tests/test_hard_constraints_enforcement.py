from __future__ import annotations

import unittest

from agent import IOSAgent
from memory.prompt_builder import PromptBuilder
from memory.protocol import ProtocolCoordinator


class _StubMemory:
    def __init__(self, constraints: list[str]) -> None:
        self.constraints = list(constraints)

    def get_profile_snapshot(self, *, user_id: str, session_id: str | None = None) -> dict:
        _ = user_id, session_id
        return {
            "hard_constraints": {
                "value": list(self.constraints),
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

    def test_protocol_extracts_and_appends_hard_constraints(self) -> None:
        protocol = ProtocolCoordinator.__new__(ProtocolCoordinator)
        patch = protocol._extract_profile_patch(
            user_message=(
                "Используй только SwiftUI, iOS 16, архитектура MVVM, "
                "без сторонних зависимостей, без рекламы, для детей."
            ),
            current_profile={"hard_constraints": ["Уже задано"]},
        )
        constraints = patch.get("hard_constraints") or []
        self.assertIn("Уже задано", constraints)
        self.assertIn("Только SwiftUI", constraints)
        self.assertIn("iOS 16", constraints)
        self.assertIn("Архитектура MVVM", constraints)
        self.assertIn("Без сторонних зависимостей", constraints)
        self.assertIn("Без рекламы", constraints)
        self.assertIn("ЦА: дети", constraints)

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

    def test_canonical_patch_keeps_existing_and_adds_protocol_constraints(self) -> None:
        protocol = ProtocolCoordinator.__new__(ProtocolCoordinator)
        patch = protocol.build_canonical_patch(
            protocol_profile={
                "experience_level": "unknown",
                "stack": "",
                "monetization_model": "",
                "target_audience": "",
                "app_idea": "",
                "current_progress": "",
                "hard_constraints": ["Только SwiftUI", "Без рекламы"],
            },
            current_profile={
                "hard_constraints": {
                    "value": ["iOS 16"],
                },
                "project_context": {
                    "value": {
                        "project_name": "",
                        "goals": [],
                        "key_decisions": [],
                    }
                },
            },
        )
        canonical_constraints = ((patch.get("hard_constraints") or {}).get("value")) or []
        self.assertIn("iOS 16", canonical_constraints)
        self.assertIn("Только SwiftUI", canonical_constraints)
        self.assertIn("Без рекламы", canonical_constraints)


if __name__ == "__main__":
    unittest.main()
