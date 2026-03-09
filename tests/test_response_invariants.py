from __future__ import annotations

import unittest

from memory.response_invariants import parse_response_markers, validate_response


class ResponseInvariantTests(unittest.TestCase):
    def test_parse_response_markers_primary_format(self) -> None:
        raw = (
            "<internal>\n"
            "[STEP_DONE: 2]\n"
            "[NEXT_STATE: EXECUTION]\n"
            "[VALIDATION_OK]\n"
            "</internal>\n"
            "<external>\n"
            "Готово, продолжаем.\n"
            "</external>"
        )
        parsed = parse_response_markers(raw)
        self.assertEqual(parsed.format, "markers")
        self.assertEqual(parsed.step_done, 2)
        self.assertEqual(parsed.next_state, "EXECUTION")
        self.assertTrue(parsed.validation_ok)
        self.assertEqual(parsed.external, "Готово, продолжаем.")

    def test_parse_response_markers_legacy_fallback(self) -> None:
        raw = "<internal>debug</internal><external>Привет!</external>"
        parsed = parse_response_markers(raw)
        self.assertEqual(parsed.format, "legacy")
        self.assertEqual(parsed.external, "Привет!")
        self.assertEqual(parsed.internal, "debug")

    def test_validate_response_rejects_forbidden_transition(self) -> None:
        parsed = parse_response_markers(
            "<internal>[STEP_DONE: 1][NEXT_STATE: DONE][VALIDATION_OK]</internal><external>ok</external>"
        )
        report = validate_response(
            parsed=parsed,
            state_object={
                "task": "t",
                "state": "EXECUTION",
                "plan": ["Шаг 1", "Шаг 2"],
                "current_step": "Шаг 2",
                "done": ["Шаг 1"],
                "artifacts": [],
                "open_questions": [],
            },
            allowed_transitions={
                "PLANNING": {"EXECUTION"},
                "EXECUTION": {"VALIDATION", "PLANNING"},
                "VALIDATION": {"DONE", "EXECUTION"},
                "DONE": set(),
            },
        ).to_dict()
        self.assertEqual(report.get("overall_status"), "fail")
        self.assertTrue(any(v.get("code") == "FORBIDDEN_TRANSITION" for v in report.get("violations") or []))

    def test_validate_response_passes_on_valid_payload(self) -> None:
        parsed = parse_response_markers(
            "<internal>[STEP_DONE: 1][NEXT_STATE: EXECUTION][VALIDATION_OK]</internal><external>ok</external>"
        )
        report = validate_response(
            parsed=parsed,
            state_object={
                "task": "t",
                "state": "EXECUTION",
                "plan": ["Шаг 1", "Шаг 2"],
                "current_step": "Шаг 2",
                "done": ["Шаг 1"],
                "artifacts": [],
                "open_questions": [],
            },
            allowed_transitions={
                "PLANNING": {"EXECUTION"},
                "EXECUTION": {"VALIDATION", "PLANNING"},
                "VALIDATION": {"DONE", "EXECUTION"},
                "DONE": set(),
            },
        ).to_dict()
        self.assertEqual(report.get("overall_status"), "ok")

    def test_unknown_next_state_is_ignored_without_failure(self) -> None:
        raw = (
            "<internal>[STEP_DONE: 1][NEXT_STATE: COLLECT_REQUIREMENTS][VALIDATION_OK]</internal>"
            "<external>ok</external>"
        )
        with self.assertLogs("memory", level="WARNING") as logs:
            parsed = parse_response_markers(raw)
        self.assertIsNone(parsed.next_state)
        self.assertEqual(parsed.next_state_raw, "COLLECT_REQUIREMENTS")
        self.assertTrue(parsed.next_state_marker_present)
        self.assertTrue(any("Неизвестный NEXT_STATE" in line for line in logs.output))

        report = validate_response(
            parsed=parsed,
            state_object={
                "task": "t",
                "state": "PLANNING",
                "plan": [],
                "current_step": None,
                "done": ["Шаг 1"],
                "artifacts": [],
                "open_questions": [],
            },
            allowed_transitions={
                "PLANNING": {"EXECUTION"},
                "EXECUTION": {"VALIDATION", "PLANNING"},
                "VALIDATION": {"DONE", "EXECUTION"},
                "DONE": set(),
            },
        ).to_dict()
        self.assertEqual(report.get("overall_status"), "ok")

    def test_done_note_is_parsed_and_logged(self) -> None:
        raw = "<internal>[STEP_DONE: 0][NEXT_STATE: PLANNING][VALIDATION_OK][DONE: план согласован, 7 блоков]</internal><external>ok</external>"
        with self.assertLogs("memory", level="INFO") as logs:
            parsed = parse_response_markers(raw)
        self.assertEqual(parsed.done_note, "план согласован, 7 блоков")
        self.assertTrue(any("done_note=план согласован, 7 блоков" in line for line in logs.output))

    def test_done_note_does_not_affect_transition_validation(self) -> None:
        raw_without_done = "<internal>[STEP_DONE: 1][NEXT_STATE: EXECUTION][VALIDATION_OK]</internal><external>ok</external>"
        raw_with_done = (
            "<internal>[STEP_DONE: 1][NEXT_STATE: EXECUTION][VALIDATION_OK]"
            "[DONE: план согласован, 7 блоков]</internal><external>ok</external>"
        )
        base_state = {
            "task": "t",
            "state": "EXECUTION",
            "plan": ["Шаг 1", "Шаг 2"],
            "current_step": "Шаг 2",
            "done": ["Шаг 1"],
            "artifacts": [],
            "open_questions": [],
        }
        allowed = {
            "PLANNING": {"EXECUTION"},
            "EXECUTION": {"VALIDATION", "PLANNING"},
            "VALIDATION": {"DONE", "EXECUTION"},
            "DONE": set(),
        }
        report_without_done = validate_response(
            parsed=parse_response_markers(raw_without_done),
            state_object=base_state,
            allowed_transitions=allowed,
        ).to_dict()
        report_with_done = validate_response(
            parsed=parse_response_markers(raw_with_done),
            state_object=base_state,
            allowed_transitions=allowed,
        ).to_dict()
        self.assertEqual(report_without_done.get("overall_status"), "ok")
        self.assertEqual(report_with_done.get("overall_status"), "ok")

    def test_parse_response_markers_extracts_open_question_and_code_artifact(self) -> None:
        raw = (
            "<internal>[STEP_DONE: 1][NEXT_STATE: EXECUTION][VALIDATION_OK]"
            "[DONE: код готов][OPEN_QUESTION: добавить тесты?][CODE_ARTIFACT]</internal>"
            "<external>```swift\nextension Array {}\n```</external>"
        )
        parsed = parse_response_markers(raw)
        self.assertEqual(parsed.done_note, "код готов")
        self.assertEqual(parsed.open_question_note, "добавить тесты?")
        self.assertEqual(parsed.code_artifact_note, "present")

        signals = parsed.to_response_signals()
        self.assertTrue(signals.has_done_phrase)
        self.assertTrue(signals.has_open_question)
        self.assertTrue(signals.has_code_artifact)

    def test_parse_response_markers_detects_fenced_code_artifact_without_marker(self) -> None:
        raw = (
            "<internal>[STEP_DONE: 1][NEXT_STATE: EXECUTION][VALIDATION_OK]</internal>"
            "<external>```swift\nextension Array {}\n```</external>"
        )
        parsed = parse_response_markers(raw)
        self.assertEqual(parsed.code_artifact_note, "fenced_code")
        self.assertTrue(parsed.to_response_signals().has_code_artifact)

    def test_planning_rejects_implementation_code(self) -> None:
        parsed = parse_response_markers(
            "<internal>[STEP_DONE: 0][NEXT_STATE: PLANNING][VALIDATION_OK]</internal>"
            "<external>extension Array { subscript(safe i: Int) -> Element? { nil } }</external>"
        )
        report = validate_response(
            parsed=parsed,
            state_object={
                "task": "t",
                "state": "PLANNING",
                "plan": [],
                "current_step": None,
                "done": [],
                "artifacts": [],
                "open_questions": [],
            },
            allowed_transitions={
                "PLANNING": {"EXECUTION"},
                "EXECUTION": {"VALIDATION", "PLANNING"},
                "VALIDATION": {"DONE", "EXECUTION"},
                "DONE": set(),
            },
        ).to_dict()
        self.assertEqual(report.get("overall_status"), "fail")
        self.assertTrue(
            any(v.get("code") == "PLANNING_CONTAINS_IMPLEMENTATION" for v in report.get("violations") or [])
        )


if __name__ == "__main__":
    unittest.main()
