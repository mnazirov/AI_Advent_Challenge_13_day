"""
context_strategies.py — 3 стратегии управления контекстом диалога.

Каждая стратегия — отдельный класс с единым интерфейсом:
    build_context(history)  → list[dict]   # что отправить в LLM
    reset()                                # сброс при новой сессии
    restore(state: dict)                   # восстановление из БД
    dump() → dict                          # сериализация в БД
    name   → str                           # идентификатор стратегии
"""

from __future__ import annotations

import json
import logging

from llm.client import LLMClient

logger = logging.getLogger("ctx_strategy")

def _create_chat_completion(client: LLMClient, **kwargs):
    """Единый вызов через абстракцию LLMClient."""
    return client.chat_completion(**kwargs)


# ══════════════════════════════════════════════════════════════════
# СТРАТЕГИЯ 1: SLIDING WINDOW
# ══════════════════════════════════════════════════════════════════

class SlidingWindowStrategy:
    """
    Самая простая стратегия.
    Держит только последние WINDOW_SIZE сообщений.
    Всё остальное молча отбрасывается.
    """

    name = "sliding_window"
    WINDOW_SIZE = 10  # последние N сообщений

    def __init__(self):
        pass  # нет состояния

    def build_context(self, history: list[dict]) -> list[dict]:
        """
        Возвращает последние WINDOW_SIZE сообщений.
        Если история короче — возвращает всю историю.
        """
        return history[-self.WINDOW_SIZE :]

    def reset(self) -> None:
        pass  # нет состояния для сброса

    def restore(self, state: dict) -> None:
        del state
        pass  # нет состояния для восстановления

    def dump(self) -> dict:
        return {}  # нечего сохранять

    def stats(self, history: list[dict]) -> dict:
        """Статистика для debug UI."""
        context = self.build_context(history)
        return {
            "strategy": self.name,
            "total_messages": len(history),
            "in_context": len(context),
            "dropped": max(0, len(history) - self.WINDOW_SIZE),
            "window_size": self.WINDOW_SIZE,
        }


# ══════════════════════════════════════════════════════════════════
# СТРАТЕГИЯ 2: STICKY FACTS / KEY-VALUE MEMORY
# ══════════════════════════════════════════════════════════════════

class StickyFactsStrategy:
    """
    Хранит структурированные факты о пользователе извлечённые из диалога.
    В контекст отправляет: блок facts + последние RECENT_N сообщений.

    Facts обновляются после каждого сообщения пользователя через LLM-вызов.
    Это дешёвый вызов (~$0.00002) но критически важный для памяти.

    Структура facts:
    {
        "goal":          "выпустить MVP iOS-приложения к июню",
        "constraints":   "минимум iOS 16, без сторонних SDK",
        "preferences":   "предпочитает SwiftUI и MVVM",
        "decisions":     ["используем async/await для сети"],
        "agreements":    ["покрывать критический путь UI-тестами"],
        "profile":       "junior iOS, pet-проект, macOS + Xcode"
    }
    """

    name = "sticky_facts"
    RECENT_N = 30  # последние N сообщений "как есть"

    # Ключи facts и их описания для LLM
    FACT_KEYS = {
        "goal": "цель проекта или разработки (явная или неявная)",
        "constraints": "ограничения: версии iOS, сроки, запреты на зависимости",
        "preferences": "предпочтения по стеку, архитектуре, стилю реализации",
        "decisions": "принятые решения (список строк)",
        "agreements": "договорённости с агентом (список строк)",
        "profile": "профиль и контекст пользователя: уровень, роль, среда разработки",
    }

    def __init__(self, client: LLMClient, model: str = "gpt-4o-mini"):
        self.client = client
        self.model = model
        self.facts: dict = {k: "" for k in self.FACT_KEYS}
        self.facts["decisions"] = []
        self.facts["agreements"] = []

    def build_context(self, history: list[dict]) -> list[dict]:
        """
        Возвращает:
        1. Блок фактов (если есть непустые)
        2. Последние RECENT_N сообщений
        """
        result = []

        facts_block = self._format_facts()
        if facts_block:
            result.append(
                {
                    "role": "system",
                    "content": facts_block,
                }
            )

        result.extend(history[-self.RECENT_N :])
        return result

    def update_facts(self, user_message: str, history: list[dict]) -> None:
        """
        Обновляет facts после сообщения пользователя.
        Вызывается в agent.py ПЕРЕД основным LLM-запросом.
        Использует последние 6 сообщений для контекста (экономия токенов).
        """
        recent_context = "\n".join(
            f"{m['role'].upper()}: {m['content'][:300]}" for m in history[-6:]
        )

        existing_facts = json.dumps(self.facts, ensure_ascii=False, indent=2)

        prompt = f"""You are an information extraction system for iOS development conversations.

CURRENT FACTS:
{existing_facts}

RECENT MESSAGES:
{recent_context}

NEW USER MESSAGE:
{user_message[:500]}

Update the facts only if the new message contains important new information.
Only update keys with new data. Keep all other keys unchanged.
For "decisions" and "agreements", append new items instead of replacing the list.

Keys and meaning:
- goal: user's iOS development goal
- constraints: constraints (target iOS version, deadlines, tooling limits)
- preferences: preferred approaches, architecture, or stack choices
- decisions: accepted decisions (list)
- agreements: agreements with the assistant (list)
- profile: user context (experience level, role, team/project setup)

Return ONLY valid JSON with the same keys. Empty values must be empty string or [].
If there is no new information, return facts unchanged."""

        try:
            response = _create_chat_completion(
                client=self.client,
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content
            updated = json.loads(raw)

            # Мержим — не заменяем целиком
            for key in self.FACT_KEYS:
                if key not in updated:
                    continue
                val = updated[key]
                if key in ("decisions", "agreements"):
                    existing = self.facts.get(key, [])
                    if isinstance(existing, list) and isinstance(val, list):
                        # Добавляем только новые элементы
                        for item in val:
                            if item and item not in existing:
                                existing.append(item)
                        self.facts[key] = existing
                    elif isinstance(val, list):
                        self.facts[key] = val
                else:
                    if val:  # не затираем непустое пустым
                        self.facts[key] = val

            tokens = response.usage.total_tokens
            logger.info("[FACTS] Обновлены: tokens=%s facts=%s", tokens, self._non_empty_count())

        except Exception as exc:
            logger.error("[FACTS] Ошибка обновления: %s", exc)
            # Безопасный fallback — факты остаются прежними

    def reset(self) -> None:
        self.facts = {k: "" for k in self.FACT_KEYS}
        self.facts["decisions"] = []
        self.facts["agreements"] = []

    def restore(self, state: dict) -> None:
        if "facts" in state and isinstance(state.get("facts"), dict):
            self.facts = state["facts"]
        logger.info("[FACTS] Восстановлены: %s непустых фактов", self._non_empty_count())

    def dump(self) -> dict:
        return {"facts": self.facts}

    def stats(self, history: list[dict]) -> dict:
        context = self.build_context(history)
        return {
            "strategy": self.name,
            "total_messages": len(history),
            "in_context": len([m for m in context if m["role"] != "system"]),
            "facts_populated": self._non_empty_count(),
            "facts": self.facts,
            "recent_n": self.RECENT_N,
        }

    # ── Приватные ──

    def _format_facts(self) -> str:
        """Форматирует факты в блок для system-сообщения."""
        lines = []
        labels = {
            "goal": "Goal",
            "constraints": "Constraints",
            "preferences": "Preferences",
            "decisions": "Decisions",
            "agreements": "Agreements",
            "profile": "Profile",
        }

        for key, label in labels.items():
            val = self.facts.get(key)
            if not val:
                continue
            if isinstance(val, list):
                if val:
                    lines.append(f"• {label}: {'; '.join(str(v) for v in val)}")
            else:
                lines.append(f"• {label}: {val}")

        if not lines:
            return ""

        return (
            "[USER MEMORY]\n"
            "Facts extracted from our conversation. Use as context:\n"
            + "\n".join(lines)
            + "\n[END USER MEMORY]"
        )

    def _non_empty_count(self) -> int:
        count = 0
        for value in self.facts.values():
            if isinstance(value, list):
                count += len(value)
            elif value:
                count += 1
        return count


# ══════════════════════════════════════════════════════════════════
# СТРАТЕГИЯ 3: BRANCHING (ветки диалога)
# ══════════════════════════════════════════════════════════════════

class BranchingStrategy:
    """
    Позволяет создавать ветки диалога от любой точки (checkpoint).

    Структура данных:
    - branches: dict[branch_id, list[messages]]  ← все ветки
    - active_branch: str                          ← текущая активная
    - checkpoints: dict[name, {branch_id, msg_index}]  ← сохранённые точки

    Ветка "main" создаётся автоматически и существует всегда.
    """

    name = "branching"

    def __init__(self):
        self.branches: dict[str, list[dict]] = {"main": []}
        self.active_branch: str = "main"
        self.checkpoints: dict[str, dict] = {}
        self._branch_counter: int = 0

    # ── Публичный интерфейс (build_context / reset / restore / dump) ──

    def build_context(self, history: list[dict]) -> list[dict]:
        """
        Возвращает сообщения текущей активной ветки.
        history игнорируется — у каждой ветки своя история.
        Возвращает ПОЛНУЮ историю активной ветки без обрезания.
        """
        del history
        branch = self.branches.get(self.active_branch, [])
        return list(branch)

    def reset(self) -> None:
        self.branches = {"main": []}
        self.active_branch = "main"
        self.checkpoints = {}
        self._branch_counter = 0

    def restore(self, state: dict) -> None:
        self.branches = state.get("branches", {"main": []})
        if "main" not in self.branches:
            self.branches["main"] = []
        self.active_branch = state.get("active_branch", "main")
        if self.active_branch not in self.branches:
            self.active_branch = "main"
        self.checkpoints = state.get("checkpoints", {})
        self._branch_counter = int(state.get("branch_counter", 0) or 0)
        logger.info(
            "[BRANCH] Восстановлено: ветки=%s активная=%s чекпоинтов=%s",
            list(self.branches.keys()),
            self.active_branch,
            len(self.checkpoints),
        )

    def dump(self) -> dict:
        return {
            "branches": self.branches,
            "active_branch": self.active_branch,
            "checkpoints": self.checkpoints,
            "branch_counter": self._branch_counter,
        }

    # ── Branch-специфичный интерфейс (вызывается из agent.py) ──

    def add_message(self, role: str, content: str) -> None:
        """
        Добавляет сообщение в текущую активную ветку.
        Вызывать вместо прямого append к conversation_history.
        """
        self.branches[self.active_branch].append({"role": role, "content": content})

    def create_checkpoint(self, name: str) -> dict:
        """
        Сохраняет текущее состояние активной ветки как checkpoint.
        От него можно будет создавать новые ветки.

        Возвращает: {"name": str, "branch": str, "msg_index": int}
        """
        branch = self.active_branch
        msgs = self.branches[branch]
        msg_idx = len(msgs)

        self.checkpoints[name] = {
            "branch": branch,
            "msg_index": msg_idx,
        }
        logger.info("[BRANCH] Checkpoint '%s': ветка=%s msg=%s", name, branch, msg_idx)
        return {"name": name, "branch": branch, "msg_index": msg_idx}

    def fork(self, checkpoint_name: str, new_branch_name: str | None = None) -> str:
        """
        Создаёт новую ветку от checkpoint.
        Новая ветка начинается с копии истории до точки checkpoint.

        Возвращает: имя новой ветки.
        """
        if checkpoint_name not in self.checkpoints:
            raise ValueError(f"Checkpoint '{checkpoint_name}' не найден")

        cp = self.checkpoints[checkpoint_name]
        source = self.branches[cp["branch"]][: cp["msg_index"]]

        # Генерируем имя новой ветки
        if not new_branch_name:
            self._branch_counter += 1
            new_branch_name = f"branch_{self._branch_counter}"

        self.branches[new_branch_name] = list(source)  # копия, не ссылка
        logger.info(
            "[BRANCH] Fork: %s → %s (%s сообщений скопировано)",
            checkpoint_name,
            new_branch_name,
            len(source),
        )
        return new_branch_name

    def switch_branch(self, branch_name: str) -> None:
        """Переключается на указанную ветку."""
        if branch_name not in self.branches:
            raise ValueError(f"Ветка '{branch_name}' не найдена")
        self.active_branch = branch_name
        logger.info("[BRANCH] Переключились на ветку: %s", branch_name)

    def delete_branch(self, branch_name: str) -> None:
        """Удаляет ветку (нельзя удалить main и активную)."""
        if branch_name == "main":
            raise ValueError("Нельзя удалить ветку main")
        if branch_name == self.active_branch:
            raise ValueError("Нельзя удалить активную ветку — сначала переключитесь")
        del self.branches[branch_name]

    def list_branches(self) -> list[dict]:
        """Список всех веток с метаданными."""
        result = []
        for name, msgs in self.branches.items():
            result.append(
                {
                    "name": name,
                    "messages": len(msgs),
                    "active": name == self.active_branch,
                }
            )
        return result

    def stats(self, history: list[dict] | None = None) -> dict:
        del history
        active_msgs = self.branches.get(self.active_branch, [])
        return {
            "strategy": self.name,
            "active_branch": self.active_branch,
            "branches": self.list_branches(),
            "checkpoints": list(self.checkpoints.keys()),
            "in_context": len(active_msgs),
            "total_messages": len(active_msgs),
        }


# ══════════════════════════════════════════════════════════════════
# СТРАТЕГИЯ 4: HISTORY COMPRESSION (rolling summary)
# ══════════════════════════════════════════════════════════════════

class HistoryCompressionStrategy:
    """
    Адаптер над существующим ContextManager из context_manager.py.
    Каждые CHUNK_SIZE сообщений старые чанки сжимаются в summary через LLM.
    В контекст отправляется: summary + последние RECENT_N сообщений.

    Отличие от Sliding Window: вместо тихого удаления — LLM-сжатие.
    Отличие от Sticky Facts: summary — связный нарратив, не key-value.
    """

    name = "history_compression"
    CHUNK_SIZE = 10
    RECENT_N = 10

    def __init__(self, client: LLMClient, model: str = "gpt-4o-mini"):
        self.client = client
        self.model = model
        self.summary: str = ""
        self.summarized_up_to = 0

    def build_context(self, history: list[dict]) -> list[dict]:
        """
        Запускает сжатие если накопился новый чанк,
        затем возвращает [summary_block] + последние RECENT_N.
        """
        self._maybe_compress(history)

        result = []
        if self.summary:
            result.append(
                {
                    "role": "system",
                    "content": (
                        "[CONVERSATION HISTORY SUMMARY]\n"
                        "Short summary of the earlier conversation:\n\n"
                        f"{self.summary}\n"
                        "[END SUMMARY]"
                    ),
                }
            )

        recent = history[-self.RECENT_N :] if len(history) > self.RECENT_N else history[:]
        result.extend(recent)
        return result

    def reset(self) -> None:
        self.summary = ""
        self.summarized_up_to = 0

    def restore(self, state: dict) -> None:
        self.summary = state.get("summary", "")
        self.summarized_up_to = int(state.get("summarized_up_to", 0) or 0)
        logger.info(
            "[COMPRESSION] Восстановлен: summary=%s симв., до сообщения #%s",
            len(self.summary),
            self.summarized_up_to,
        )

    def dump(self) -> dict:
        return {
            "summary": self.summary,
            "summarized_up_to": self.summarized_up_to,
        }

    def stats(self, history: list[dict]) -> dict:
        return {
            "strategy": self.name,
            "total_messages": len(history),
            "in_context": min(len(history), self.RECENT_N),
            "summarized_up_to": self.summarized_up_to,
            "summary_len": len(self.summary),
            "summary": self.summary,
            "chunk_size": self.CHUNK_SIZE,
            "recent_n": self.RECENT_N,
        }

    # ── Приватные ──

    def _maybe_compress(self, history: list[dict]) -> None:
        if len(history) <= self.RECENT_N:
            return

        safe_end = len(history) - self.RECENT_N
        next_chunk_end = ((self.summarized_up_to // self.CHUNK_SIZE) + 1) * self.CHUNK_SIZE

        if next_chunk_end > safe_end:
            return  # чанк ещё не накопился или уходит в recent

        chunk = history[self.summarized_up_to : next_chunk_end]
        if not chunk:
            return

        new_summary = self._compress_chunk(chunk)
        if new_summary:
            self.summary = new_summary
            self.summarized_up_to = next_chunk_end
            logger.info(
                "[COMPRESSION] Сжат чанк [%s:%s]: summary=%s симв.",
                self.summarized_up_to - self.CHUNK_SIZE,
                self.summarized_up_to,
                len(self.summary),
            )

    def _compress_chunk(self, chunk: list[dict]) -> str:
        chunk_text = "\n".join(f"{m['role'].upper()}: {m['content'][:400]}" for m in chunk)

        if self.summary:
            prompt = (
                f"CURRENT SUMMARY:\n{self.summary}\n\n"
                f"NEW MESSAGES:\n{chunk_text}\n\n"
                "Update the summary with important facts from the new messages. "
                "Keep key numbers, goals, and decisions. "
                "Remove outdated details. Maximum 8 sentences. Plain text only."
            )
        else:
            prompt = (
                f"CONVERSATION MESSAGES:\n{chunk_text}\n\n"
                "Create a concise summary: goals, numbers, decisions, context. "
                "Maximum 8 sentences. Plain text only."
            )

        try:
            response = _create_chat_completion(
                client=self.client,
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=350,
                temperature=0,
            )
            result = (response.choices[0].message.content or "").strip()
            logger.info("[COMPRESSION] tokens=%s", response.usage.total_tokens)
            return result
        except Exception as exc:
            logger.error("[COMPRESSION] Ошибка LLM: %s", exc)
            return ""  # безопасный fallback


# ══════════════════════════════════════════════════════════════════
# МЕНЕДЖЕР СТРАТЕГИЙ — единая точка входа
# ══════════════════════════════════════════════════════════════════

STRATEGY_NAMES = ["sliding_window", "sticky_facts", "branching", "history_compression"]


class ContextStrategyManager:
    """
    Хранит все три стратегии и знает какая активна.
    Это единственный класс с которым работает agent.py.
    """

    def __init__(self, client: LLMClient, model: str = "gpt-4o-mini"):
        self.model = model
        self._strategies = {
            "sliding_window": SlidingWindowStrategy(),
            "sticky_facts": StickyFactsStrategy(client=client, model=model),
            "branching": BranchingStrategy(),
            "history_compression": HistoryCompressionStrategy(client=client, model=model),
        }
        self._active: str = "sticky_facts"  # дефолт

    @property
    def active(self) -> str:
        return self._active

    @property
    def strategy(self):
        """Возвращает текущую активную стратегию."""
        return self._strategies[self._active]

    def set_strategy(self, name: str) -> None:
        """Переключает стратегию. Вызывается из /debug/ctx-strategy."""
        if name not in self._strategies:
            raise ValueError(f"Неизвестная стратегия: {name}. Доступны: {STRATEGY_NAMES}")
        self._active = name
        logger.info("[CTX] Стратегия переключена на: %s", name)

    def set_model(self, model: str) -> None:
        """Обновляет модель у стратегий, которые обращаются к LLM."""
        self.model = model
        for strategy in self._strategies.values():
            if hasattr(strategy, "model"):
                strategy.model = model
        logger.info("[CTX] Модель обновлена: %s", model)

    def build_context(self, history: list[dict]) -> list[dict]:
        """Делегирует активной стратегии."""
        return self.strategy.build_context(history)

    def reset_all(self) -> None:
        """Сбрасывает все стратегии (при новой сессии)."""
        for strategy in self._strategies.values():
            strategy.reset()
        self._active = "sticky_facts"

    def restore(self, state: dict) -> None:
        """
        Восстанавливает состояние из БД.
        state: {"active": "sticky_facts", "sliding_window": {}, "sticky_facts": {...}, ...}
        """
        self._active = state.get("active", "sticky_facts")
        if self._active not in self._strategies:
            self._active = "sticky_facts"
        for name, strategy in self._strategies.items():
            if name in state and isinstance(state.get(name), dict):
                strategy.restore(state[name])

    def dump(self) -> dict:
        """Сериализует всё для сохранения в БД."""
        result = {"active": self._active}
        for name, strategy in self._strategies.items():
            result[name] = strategy.dump()
        return result

    def stats(self, history: list[dict]) -> dict:
        """Статистика активной стратегии для debug UI."""
        if self._active == "branching":
            return self.strategy.stats()
        return self.strategy.stats(history)
