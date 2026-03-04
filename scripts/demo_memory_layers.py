from __future__ import annotations

import argparse
import os
import textwrap
from typing import Any
from uuid import uuid4

import storage


def _load_env_if_needed() -> None:
    """Loads OPENAI_API_KEY from project .env if not already set."""
    if os.getenv("OPENAI_API_KEY"):
        return
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env_path = os.path.join(root_dir, ".env")
    if not os.path.exists(env_path):
        return
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                if key.strip() == "OPENAI_API_KEY":
                    os.environ["OPENAI_API_KEY"] = value.strip().strip('"').strip("'")
                    return
    except Exception:
        return


def _print_header(title: str) -> None:
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)


def _log_action(action: str, detail: str = "") -> None:
    print(f"  ACTION : {action}" + (f" → {detail}" if detail else ""))


def _log_write(layer: str, summary: str) -> None:
    print(f"  WRITE  → {layer:<10} : {summary}")


def _log_read(layer: str, summary: str) -> None:
    print(f"  READ   ← {layer:<10} : {summary}")


def _log_result(allowed: bool, reason: str = "") -> None:
    status = "allowed ✓" if allowed else "blocked ✗"
    print(f"  RESULT : {status}" + (f" ({reason})" if reason else ""))


def _print_scene(label: str, status: str = "START") -> None:
    marker = "┌" if status == "START" else "└"
    tail_len = max(0, 60 - len(label))
    print(f"\n{marker}── [SCENE] {label} {'─' * tail_len}")


def _format_short_term_snapshot(short_messages: list[dict[str, str]], limit_n: int = 30) -> list[str]:
    turns = short_messages[-max(1, int(limit_n)) :]
    lines = [f"[SHORT_TERM_SNAPSHOT] last_turns={len(turns)}"]
    for msg in turns:
        role = str(msg.get("role") or "")
        content = str(msg.get("content") or "")[:80]
        lines.append(f"  - {role}: {content}")
    return lines


def _print_working_snapshot(working) -> None:
    if not working:
        print("  [WORKING] none")
        return
    print(f"  [WORKING] state        = {working.state.value}")
    print(f"  [WORKING] current_step = {working.current_step!r}")
    print(f"  [WORKING] plan         = {working.plan}")
    print(f"  [WORKING] done_steps   = {working.done_steps}")
    print(f"  [WORKING] open_q       = {working.open_questions}")


def _print_layers(agent: Any, session_id: str, user_id: str) -> None:
    short = agent.memory.short_term.get_context(session_id)
    working = agent.memory.working.load(session_id)
    longterm = agent.memory.long_term.retrieve(user_id=user_id, query="", top_k=3)

    for line in _format_short_term_snapshot(short, limit_n=getattr(agent.memory.short_term, "limit_n", 30)):
        print(line)

    _print_working_snapshot(working)

    profile = longterm.get("profile") or {}
    print("[LONG_TERM_SNAPSHOT][PROFILE]")
    if profile:
        print(f"  style: {profile.get('style')}")
        print(f"  constraints: {(profile.get('constraints') or [])[:3]}")
        print(f"  context: {(profile.get('context') or [])[:3]}")
        print(f"  tags: {(profile.get('tags') or [])[:5]}")
    else:
        print("  none")

    print("[LONG_TERM_SNAPSHOT][DECISIONS_TOP_K]")
    decisions = longterm.get("decisions") or []
    if decisions:
        for d in decisions:
            title = str(d.get("text") or "").replace("\n", " ").strip()[:80]
            print(f"  - [{d.get('id')}] {title} | tags={d.get('tags') or []}")
    else:
        print("  none")

    print("[LONG_TERM_SNAPSHOT][NOTES_TOP_K]")
    notes = longterm.get("notes") or []
    if notes:
        for n in notes:
            title = str(n.get("text") or "").replace("\n", " ").strip()[:80]
            print(f"  - [{n.get('id')}] {title} | tags={n.get('tags') or []}")
    else:
        print("  none")


def _print_prompt_preview(agent: Any) -> None:
    preview = agent.last_prompt_preview or {}
    print("[PROMPT_PREVIEW]")
    for key in ["section_profile", "section_decisions", "section_working", "section_short_term"]:
        if key in preview:
            print(f"  {preview[key]}")


def _print_reply(reply: str) -> None:
    print("[ASSISTANT_REPLY]")
    print(textwrap.shorten(reply.replace("\n", " "), width=500, placeholder=" ..."))


def _print_demo_conclusions() -> None:
    print("\n[DEMO CONCLUSIONS]")
    print("- Short-term: показано bounded окно N и исчезновение ранних реплик.")
    print("- Working: в PLANNING без шагов блокируется запрос на финальное решение; в EXECUTION ответ разрешается.")
    print("- Long-term: профиль/решения сохраняются между сессиями и влияют на стиль ответа.")
    print("- Prompt impact: в preview видны секции [PROFILE]/[DECISIONS]/[WORKING]/[SHORT_TERM].")


def run_demo(real_llm: bool = True) -> None:
    from agent import IOSAgent
    from llm.mock_client import MockLLMClient

    storage.init_db()
    session_id = storage.create_session()
    user_id = f"demo_user_{uuid4().hex[:8]}"
    _load_env_if_needed()

    if real_llm and not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required for --real-llm mode")

    agent = IOSAgent(model="gpt-5-mini")
    if not real_llm:
        agent.llm_client = MockLLMClient()

    scene_a_label = "SCENE A: SHORT-TERM MEMORY WINDOW (iOS КОНТЕКСТ)"
    _print_header(scene_a_label)
    _print_scene(scene_a_label, "START")
    _log_action("clear_session", f"session_id={session_id}")
    agent.memory.clear_session(session_id)
    for i in range(1, 36):
        month = ((i - 1) % 12) + 1
        agent.memory.short_term.append(
            session_id,
            "user",
            f"Заметка по спринту #{i}: в итерации {month} добавили новый SwiftUI-экран и фиксы",
        )
    _log_write("SHORT", f"appended {i} turns")
    _print_layers(agent, session_id, user_id)
    _log_read("SHORT", "last N turns loaded into prompt")
    reply = agent.chat(
        "Учитывай последние реплики и кратко резюмируй мой iOS контекст",
        session_id=session_id,
        user_id=user_id,
    )
    _print_prompt_preview(agent)
    _print_reply(reply)
    _print_scene(scene_a_label, "END")

    scene_b_label = "SCENE B: WORKING MEMORY + STATE MACHINE (ПЛАН iOS-ФИЧИ)"
    _print_header(scene_b_label)
    _print_scene(scene_b_label, "START")
    goal = "Собрать план реализации экрана авторизации на SwiftUI"
    agent.memory.working.start_task(session_id=session_id, goal=goal)
    _log_action("start_task", f'goal="{goal}"')
    _log_write("WORKING", "state=PLANNING, plan=[], current_step=None")
    blocked = agent.chat("Сразу дай финальный код экрана авторизации", session_id=session_id, user_id=user_id)
    _log_result(False, "state=PLANNING, no steps defined")
    _print_layers(agent, session_id, user_id)
    _print_reply(blocked)

    plan = [
        "Собрать требования и сценарии авторизации",
        "Смоделировать архитектуру экрана (MVVM + Coordinator)",
        "Реализовать UI, валидацию и обработку ошибок",
    ]
    current_step = "Собрать требования и сценарии авторизации"
    ctx = agent.memory.working.update(
        session_id,
        plan=plan,
        current_step=current_step,
    )
    agent.memory.working.transition_state(ctx, ctx.state.__class__("EXECUTION"))
    agent.memory.working.save(ctx)
    _log_write("WORKING", f"plan=[{len(plan)} steps], current_step={current_step!r}")
    _log_action("transition_state", "PLANNING → EXECUTION")

    allowed = agent.chat("Теперь составь пошаговый план реализации", session_id=session_id, user_id=user_id)
    _log_read("WORKING", "state=EXECUTION → unblocked")
    _log_result(True)
    _print_layers(agent, session_id, user_id)
    _print_prompt_preview(agent)
    _print_reply(allowed)
    _print_scene(scene_b_label, "END")

    scene_c_label = "SCENE C: LONG-TERM MEMORY ACROSS RESTART (ПРАВИЛА iOS-ПРОЕКТА)"
    _print_header(scene_c_label)
    _print_scene(scene_c_label, "START")
    agent.chat(
        "С этого момента всегда отвечай кратко, с примерами Swift-кода и без объяснения базовых терминов. "
        "Решили использовать MVVM + Coordinator.",
        session_id=session_id,
        user_id=user_id,
    )
    _log_write("LONG", "profile style + decision saved")
    _print_layers(agent, session_id, user_id)

    print("\n[RESTART] Creating a fresh agent instance...")
    _log_action("restart", "new agent instance, same user_id")
    agent2 = IOSAgent(model="gpt-5-mini")
    if not real_llm:
        agent2.llm_client = MockLLMClient()

    new_session = storage.create_session()
    reply_after_restart = agent2.chat(
        "Какой формат рекомендаций и какие архитектурные стандарты мы используем?",
        session_id=new_session,
        user_id=user_id,
    )
    _log_read("LONG", "profile + decisions retrieved")
    _log_result(True, "long-term context applied")
    _print_layers(agent2, new_session, user_id)
    _print_prompt_preview(agent2)
    _print_reply(reply_after_restart)
    _print_scene(scene_c_label, "END")
    _print_demo_conclusions()

    print("\nDemo completed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Demo memory layers for video")
    parser.add_argument("--real-llm", action="store_true", help="Use real OpenAI LLM (default)")
    parser.add_argument("--mock", action="store_true", help="Use deterministic mock client")
    args = parser.parse_args()

    real_llm = True
    if args.mock:
        real_llm = False
    elif args.real_llm:
        real_llm = True

    run_demo(real_llm=real_llm)


if __name__ == "__main__":
    main()
