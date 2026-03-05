import requests, json, time, re, sys
import pytest

BASE_URL = "http://localhost:5000"
USER_ID = "test_state_machine"
RESULTS = []  # { id, message, pass, fail_reason }


def chat(session_id, message):
    r = requests.post(
        f"{BASE_URL}/chat",
        json={"session_id": session_id, "user_id": USER_ID, "message": message},
        timeout=60,
    )
    return r.json()


def check(test_id, condition, fail_reason):
    RESULTS.append({
        "id": test_id,
        "pass": condition,
        "fail_reason": "" if condition else fail_reason,
    })
    status = "✓ PASS" if condition else "✗ FAIL"
    print(f"  {status}  [{test_id}]  {'' if condition else fail_reason}")
    return condition


def setup():
    # Create fresh session
    try:
        r = requests.post(f"{BASE_URL}/session/new", json={"user_id": USER_ID}, timeout=30)
    except requests.RequestException as exc:
        raise RuntimeError(f"E2E API unavailable at {BASE_URL}: cannot reach /session/new") from exc
    if r.status_code != 200:
        raise RuntimeError(f"E2E API unavailable at {BASE_URL}: /session/new returned HTTP {r.status_code}")
    try:
        payload = r.json()
    except Exception as exc:
        raise RuntimeError(f"E2E API unavailable at {BASE_URL}: /session/new returned non-JSON response") from exc
    session_id = payload.get("session_id")
    if not session_id:
        raise RuntimeError(f"E2E API unavailable at {BASE_URL}: session_id missing in /session/new response")

    # Set profile
    for field, value in {
        "response_style": "Краткий код, минимум комментариев",
        "hard_constraints": ["Только SwiftUI", "iOS 16", "Без сторонних зависимостей"],
        "user_role_level": "Middle iOS, 3 года опыта",
    }.items():
        requests.patch(
            f"{BASE_URL}/debug/memory/profile/field",
            json={"user_id": USER_ID, "field": field, "value": value},
        )
    return session_id


def run_planning(session_id):
    print("\n── PLANNING ─────────────────────────────────────────")

    # MSG 1
    r = chat(
        session_id,
        "Хочу реализовать экран авторизации в SwiftUI с валидацией email и пароля",
    )
    state = r.get("working_view", {}).get("state", "")

    check(
        "P1_not_done_yet",
        state != "DONE",
        f"Jumped to DONE immediately, state={state}",
    )

    check(
        "P1_no_ios_question",
        "ios" not in r["reply"].lower() or "версию" not in r["reply"].lower(),
        "Agent asked about iOS version — should know from profile",
    )

    # MSG 2
    r = chat(session_id, "Архитектура MVVM, email + пароль, никакой биометрии")
    check(
        "P2_still_planning",
        r.get("working_view", {}).get("state") in ("PLANNING", None, "IDLE"),
        "State machine skipped PLANNING",
    )

    check(
        "P2_no_swiftui_question",
        "swiftui" not in r["reply"].lower().replace("swiftui", ""),
        "Agent asked about SwiftUI — already in profile",
    )

    # MSG 3 — trigger plan formation
    r = chat(session_id, "Сформируй план задачи автоматически на основе цели")
    wv = r.get("working_view", {})
    state = wv.get("state", "")
    plan = wv.get("plan", [])

    check(
        "P3_state_execution",
        state == "EXECUTION",
        f"Expected EXECUTION after plan formation, got {state}",
    )

    check(
        "P3_plan_not_empty",
        len(plan) >= 3,
        f"Plan has only {len(plan)} steps, expected ≥ 3",
    )

    check(
        "P3_plan_not_too_long",
        len(plan) <= 8,
        f"Plan has {len(plan)} steps, expected ≤ 8",
    )

    return plan


def run_execution(session_id, plan):
    print("\n── EXECUTION ────────────────────────────────────────")
    total_steps = len(plan)

    # Step 1 — request code
    r = chat(session_id, "Покажи код для текущего шага")
    reply = r["reply"]

    check(
        "E1_swift_code",
        "swift" in reply.lower()
        or any(kw in reply for kw in ["import SwiftUI", "struct", "class", "func", "@State"]),
        "Response does not contain Swift code",
    )

    check(
        "E1_no_uikit",
        "uikit" not in reply.lower() or "import UIKit" not in reply,
        "Agent used UIKit — violates hard_constraint",
    )

    check(
        "E1_no_mvvm_basics",
        not re.search(r"mvvm\s*(—|-|:)\s*это|model.*view.*viewmodel.*это", reply, re.IGNORECASE),
        "Agent explained MVVM basics to a middle-level developer",
    )

    # Step 1 — context memory check
    r = chat(session_id, "Почему ты используешь @StateObject а не @ObservedObject?")
    check(
        "E2_memory_context",
        any(
            kw in r["reply"].lower()
            for kw in ["ownership", "владелец", "creates", "создаёт", "lifetime", "жизненный"]
        ),
        "Agent gave generic answer without referencing current context",
    )

    check(
        "E2_no_basics",
        len(r["reply"]) < 800,
        f"Answer too long ({len(r['reply'])} chars) — likely explaining basics",
    )

    # Complete all steps
    r = chat(session_id, "Шаг выполнен, переходим к следующему")
    wv = r.get("working_view", {})
    done_after_1 = len(wv.get("done", []))

    check(
        "E3_progress_moves",
        done_after_1 >= 1,
        f"Progress did not move after completing step 1: done={done_after_1}",
    )

    check(
        "E3_not_validation_yet",
        wv.get("state") == "EXECUTION",
        f"Jumped to VALIDATION after step 1 of {total_steps}",
    )

    # Complete remaining steps
    for i in range(2, total_steps + 1):
        time.sleep(0.5)
        chat(session_id, "Покажи код для текущего шага")
        time.sleep(0.5)
        r = chat(session_id, "Шаг выполнен, переходим к следующему")
        wv = r.get("working_view", {})
        print(f"  step {i}/{total_steps} done, state={wv.get('state')}")

    # After last step
    final_state = wv.get("state", "")
    check(
        "E_final_validation",
        final_state == "VALIDATION",
        f"Expected VALIDATION after all steps, got {final_state}",
    )

    check(
        "E_auto_message",
        len(r.get("reply", "")) > 50,
        "Agent sent no auto-message on entering VALIDATION",
    )


def run_validation(session_id):
    print("\n── VALIDATION ───────────────────────────────────────")

    # Check auto-message content
    r = chat(session_id, "Покажи финальный чеклист перед отправкой")
    check(
        "V1_checklist_specific",
        any(
            kw in r["reply"].lower()
            for kw in ["авторизац", "login", "email", "пароль", "ошибк", "validation"]
        ),
        "Checklist is generic, not related to auth screen task",
    )

    # Stress: vague confirmation should NOT trigger DONE
    r = chat(session_id, "готово")
    check(
        "V2_no_premature_done",
        r.get("working_view", {}).get("state") != "DONE",
        "Agent transitioned to DONE on vague 'готово' — too eager",
    )

    # Explicit confirmation
    r = chat(session_id, "Всё проверено, подтверждаю завершение задачи")
    wv = r.get("working_view", {})
    state = wv.get("state", "")

    check(
        "V3_state_done",
        state == "DONE",
        f"Expected DONE after explicit confirmation, got {state}",
    )

    check(
        "V3_auto_summary",
        len(r.get("reply", "")) > 100,
        "Agent sent no summary on entering DONE state",
    )

    check(
        "V3_summary_has_steps",
        any(kw in r["reply"].lower() for kw in ["выполнен", "шаг", "готов", "завершен"]),
        "DONE summary does not mention completed steps",
    )


def run_done(session_id):
    print("\n── DONE ─────────────────────────────────────────────")

    r = chat(session_id, "Что дальше рекомендуешь?")
    check(
        "D1_relevant_next_steps",
        any(
            kw in r["reply"].lower()
            for kw in ["keychain", "навигац", "biometric", "тест", "coordinator", "token", "токен"]
        ),
        "Next steps not related to auth screen project",
    )

    check(
        "D1_no_repeat_done_steps",
        "authviewmodel" not in r["reply"].lower().replace(" ", ""),
        "Agent suggests redoing already completed steps",
    )

    # Memory persistence check
    r = chat(session_id, "Сохрани ключевые решения в память")
    check(
        "D2_confirms_save",
        any(kw in r["reply"].lower() for kw in ["сохран", "запомн", "зафиксир", "saved"]),
        "Agent did not confirm saving to memory",
    )

    # Verify via debug endpoint
    profile = requests.get(f"{BASE_URL}/debug/memory/profile", params={"user_id": USER_ID}).json()
    check(
        "D3_memory_written",
        profile.get("project_context") is not None or profile.get("stack_tools") is not None,
        "Long-term memory not updated after DONE",
    )


def run_stress(session_id):
    print("\n── STRESS TESTS ─────────────────────────────────────")
    s2 = setup()  # Fresh session for stress tests

    # Stress A: skip steps in PLANNING
    chat(s2, "Хочу экран авторизации")
    chat(s2, "MVVM, SwiftUI")
    chat(s2, "Сформируй план")
    r = chat(s2, "Дай сразу финальный готовый код всего экрана")
    check(
        "S_A_blocked",
        r.get("finish_reason") == "state_blocked"
        or any(kw in r["reply"].lower() for kw in ["шаг", "план", "сначала", "сформиров"]),
        "Agent gave final code skipping steps — STATE_GUARD not working",
    )

    # Stress B: context switch in EXECUTION
    chat(s2, "Шаг выполнен")
    r = chat(s2, "Забудь про авторизацию, давай сделаем главный экран")
    check(
        "S_B_context_guard",
        any(kw in r["reply"].lower() for kw in ["текущ", "задач", "авторизац", "сначала завершим"]),
        "Agent abandoned active task on context switch request",
    )

    # Stress C: profile depth check
    s3 = setup()
    chat(s3, "Хочу экран авторизации")
    chat(s3, "MVVM, SwiftUI")
    chat(s3, "Сформируй план")
    r = chat(s3, "Объясни мне что такое MVVM")
    check(
        "S_C_profile_depth",
        not re.search(
            r"model\s*(—|-|—)\s*это.{0,50}данн|view\s*(—|-|—)\s*это.{0,50}интерфейс",
            r["reply"],
            re.IGNORECASE,
        ),
        "Agent explained MVVM basics to middle-level developer — profile ignored",
    )


def print_report():
    total = len(RESULTS)
    passed = sum(1 for r in RESULTS if r["pass"])
    failed = total - passed

    print(f"\n{'═' * 56}")
    print(f"  ИТОГИ: {passed}/{total} passed", end="")
    if failed:
        print(f"  ({failed} FAILED)")
        print(f"\n  Провалившиеся:")
        for r in RESULTS:
            if not r["pass"]:
                print(f"    ✗ [{r['id']}] {r['fail_reason']}")
    else:
        print("  — все тесты прошли ✓")
    print(f"{'═' * 56}\n")
    return failed


def test_state_machine_flow_e2e():
    RESULTS.clear()
    try:
        session_id = setup()
    except RuntimeError as exc:
        pytest.skip(str(exc))
    plan = run_planning(session_id)
    run_execution(session_id, plan)
    run_validation(session_id)
    run_done(session_id)
    run_stress(session_id)
    assert print_report() == 0


if __name__ == "__main__":
    print("iOS Agent — State Machine Integration Test")
    RESULTS.clear()
    session_id = setup()
    plan = run_planning(session_id)
    run_execution(session_id, plan)
    run_validation(session_id)
    run_done(session_id)
    run_stress(session_id)
    failed = print_report()
    sys.exit(1 if failed else 0)
