from __future__ import annotations

import unittest

from agent import IOSAgent


class ModelRegistryTests(unittest.TestCase):
    def test_default_model_is_gpt_5_3_instant(self) -> None:
        self.assertEqual(IOSAgent.DEFAULT_MODEL, "gpt-5.3-instant")

    def test_gpt_5_3_instant_is_available(self) -> None:
        models = IOSAgent.available_models()
        self.assertIn("gpt-5.3-instant", models)

    def test_validate_accepts_gpt_5_3_instant(self) -> None:
        validated = IOSAgent._validate_model("gpt-5.3-instant")
        self.assertEqual(validated, "gpt-5.3-instant")


if __name__ == "__main__":
    unittest.main()
