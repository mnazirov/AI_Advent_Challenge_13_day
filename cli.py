#!/usr/bin/env python3
"""
CLI-интерфейс для iOS AI-ассистента.
Запуск: python3 cli.py
Требует запущенного сервера: python3 app.py
"""

from __future__ import annotations

import json
import os
import sys
import textwrap
import uuid
from dataclasses import dataclass, field
from typing import Any

import requests


# ── Конфигурация ─────────────────────────────────────────────────────────────

_PORT = os.environ.get("PORT", "5000")
BASE_URL = f"http://localhost:{_PORT}"  # читает PORT из env как app.py
WIDTH = 80  # ширина терминала для wrap

# USER_ID персистентен между запусками — хранится рядом с cli.py
_USER_ID_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cli_user_id")


def _load_or_create_user_id() -> str:
    try:
        if os.path.exists(_USER_ID_FILE):
            saved = open(_USER_ID_FILE, "r", encoding="utf-8").read().strip()
            if saved:
                return saved
    except OSError:
        pass

    uid = f"cli_user_{uuid.uuid4().hex[:8]}"
    try:
        open(_USER_ID_FILE, "w", encoding="utf-8").write(uid)
    except OSError:
        # Если запись файла не удалась — продолжаем с runtime user_id.
        pass
    return uid


# ── Модели данных ─────────────────────────────────────────────────────────────


@dataclass
class CLIState:
    """Текущее состояние CLI-сессии. Обновляется после каждого ответа."""

    model: str = "—"
    provider: str = "—"
    ctx_strategy: str = "—"
    session_id: str = ""
    project_id: str = ""
    project_name: str = "—"
    working_state: str = "IDLE"
    short_term_turns: int = 0


@dataclass
class APIResponse:
    ok: bool
    data: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    status_code: int = 0


# ── HTTP-клиент ───────────────────────────────────────────────────────────────


class APIClient:
    """
    Тонкая обёртка над requests.
    Все URL собираются здесь. Нигде больше BASE_URL не упоминается.
    """

    def __init__(self, base_url: str, user_id: str):
        self._base = base_url.rstrip("/")
        self._user = user_id
        self._session = requests.Session()

    @property
    def user_id(self) -> str:
        return self._user

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        body: dict[str, Any] | None = None,
        timeout: int = 60,
    ) -> APIResponse:
        url = f"{self._base}{path}"
        try:
            response = self._session.request(
                method=method,
                url=url,
                params=params,
                json=body,
                timeout=timeout,
            )
        except requests.RequestException as exc:
            return APIResponse(ok=False, error=str(exc))

        payload: Any = {}
        if response.text:
            try:
                payload = response.json()
            except ValueError:
                payload = {}

        if not isinstance(payload, dict):
            payload = {"data": payload}

        if response.ok:
            return APIResponse(ok=True, data=payload, status_code=response.status_code)

        error_msg = (
            str(payload.get("error") or payload.get("message") or "").strip()
            or response.text.strip()
            or f"HTTP {response.status_code}"
        )
        return APIResponse(ok=False, data=payload, error=error_msg, status_code=response.status_code)

    def _get(self, path: str, params: dict[str, Any] | None = None, timeout: int = 60) -> APIResponse:
        return self._request("GET", path, params=params, timeout=timeout)

    def _post(self, path: str, body: dict[str, Any] | None = None, timeout: int = 120) -> APIResponse:
        return self._request("POST", path, body=body or {}, timeout=timeout)

    def _patch(self, path: str, body: dict[str, Any] | None = None, timeout: int = 30) -> APIResponse:
        return self._request("PATCH", path, body=body or {}, timeout=timeout)

    def _delete(self, path: str, body: dict[str, Any] | None = None, timeout: int = 30) -> APIResponse:
        return self._request("DELETE", path, body=body or {}, timeout=timeout)

    # ── Модели ────────────────────────────────────────────────────────────────
    def get_models(self) -> APIResponse:
        return self._get("/models")

    def set_model(self, model: str) -> APIResponse:
        return self._post("/model", {"model": model})

    # ── Проекты ───────────────────────────────────────────────────────────────
    def get_projects(self) -> APIResponse:
        resp = self._get("/projects", {"user_id": self._user})
        if not resp.ok:
            return resp

        projects = resp.data.get("projects")
        if projects is None:
            raw = resp.data.get("data")
            if isinstance(raw, list):
                projects = raw
        if not isinstance(projects, list):
            return APIResponse(ok=False, error="Некорректный формат ответа /projects", status_code=resp.status_code)
        return APIResponse(ok=True, data={"projects": projects}, status_code=resp.status_code)

    def create_project(self, name: str) -> APIResponse:
        return self._post("/projects", {"user_id": self._user, "name": name})

    def activate_project(self, project_id: str) -> APIResponse:
        return self._patch(f"/projects/{project_id}/activate", {"user_id": self._user})

    def delete_project(self, project_id: str) -> APIResponse:
        return self._delete(f"/projects/{project_id}", {"user_id": self._user})

    # ── Сессия ────────────────────────────────────────────────────────────────
    def restore_session(self) -> APIResponse:
        return self._get("/session/restore", {"user_id": self._user})

    def reset_session(self, session_id: str) -> APIResponse:
        return self._post("/reset", {"user_id": self._user, "session_id": session_id})

    # ── Чат ───────────────────────────────────────────────────────────────────
    def chat(
        self,
        message: str,
        session_id: str,
        project_id: str,
        client_intent: dict[str, Any] | None = None,
    ) -> APIResponse:
        body: dict[str, Any] = {
            "user_id": self._user,
            "session_id": session_id,
            "project_id": project_id,
            "message": message,
        }
        if client_intent:
            body["client_intent"] = client_intent
        return self._post("/chat", body)

    def chat_with_intent(self, message: str, intent: str, session_id: str, project_id: str) -> APIResponse:
        """Shortcut для отправки сообщения с явным client_intent."""
        return self.chat(
            message=message,
            session_id=session_id,
            project_id=project_id,
            client_intent={"intent": intent, "payload": {}},
        )

    # ── Контекст-стратегия ────────────────────────────────────────────────────
    def set_ctx_strategy(self, strategy: str, session_id: str) -> APIResponse:
        return self._post(
            "/debug/ctx-strategy",
            {
                "user_id": self._user,
                "session_id": session_id,
                "strategy": strategy,
            },
        )

    # ── Память ────────────────────────────────────────────────────────────────
    def get_memory_layers(self, session_id: str) -> APIResponse:
        return self._get(
            "/debug/memory-layers",
            {
                "user_id": self._user,
                "session_id": session_id,
            },
        )

    def clear_memory(self, layer: str, session_id: str) -> APIResponse:
        return self._post(
            f"/debug/memory/{layer}/clear",
            {
                "user_id": self._user,
                "session_id": session_id,
            },
        )

    # ── Профиль ───────────────────────────────────────────────────────────────
    def get_profile(self, session_id: str) -> APIResponse:
        return self._get(
            "/debug/memory/profile",
            {
                "user_id": self._user,
                "session_id": session_id,
            },
        )

    # ── MCP ───────────────────────────────────────────────────────────────────
    def get_mcp_tools(self) -> APIResponse:
        return self._get("/mcp/time/tools")

    def call_mcp_tool(self, tool: str, arguments: dict[str, Any] | None = None) -> APIResponse:
        return self._post(
            "/mcp/time/call",
            {
                "tool": tool,
                "arguments": arguments or {},
            },
        )

    # ── Ветвление ─────────────────────────────────────────────────────────────
    def checkpoint(self, session_id: str, name: str) -> APIResponse:
        return self._post(
            "/ctx/checkpoint",
            {
                "user_id": self._user,
                "session_id": session_id,
                "name": name,
            },
        )

    def fork(self, session_id: str, checkpoint: str, branch_name: str | None = None) -> APIResponse:
        body: dict[str, Any] = {
            "user_id": self._user,
            "session_id": session_id,
            "checkpoint": checkpoint,
        }
        if branch_name:
            body["branch_name"] = branch_name
        return self._post("/ctx/fork", body)

    def switch_branch(self, session_id: str, branch: str) -> APIResponse:
        return self._post(
            "/ctx/switch-branch",
            {
                "user_id": self._user,
                "session_id": session_id,
                "branch": branch,
            },
        )


# ── Рендерер ─────────────────────────────────────────────────────────────────


class Renderer:
    """Всё что касается вывода в терминал — только здесь."""

    SEP = "─" * WIDTH
    SEP2 = "═" * WIDTH

    @staticmethod
    def _wrap(text: str, indent: int = 0) -> str:
        prefix = " " * indent
        return textwrap.fill(text, width=WIDTH, initial_indent=prefix, subsequent_indent=prefix)

    def header(self, state: CLIState) -> None:
        print(self.SEP2)
        print(" iOS AI-ассистент · Swift/SwiftUI · локальный режим")
        print(self.SEP)
        print(f"  Модель:           {state.model}")
        print(f"  Провайдер:        {state.provider}")
        print(f"  Стратегия ctx:    {state.ctx_strategy}")
        print(f"  Проект:           {state.project_name}")
        print(f"  Состояние задачи: {state.working_state}")
        print(self.SEP)
        print("  Введите сообщение или /help для списка команд")
        print(self.SEP2)

    def assistant(self, text: str, meta: dict[str, Any]) -> None:
        shown = text if str(text or "").strip() else "(пустой ответ)"
        print(f"\n╭─ Ассистент [{meta.get('working_state', '?')}] ─")
        for line in shown.split("\n"):
            wrapped = self._wrap(line, indent=2) if line.strip() else ""
            print(wrapped or "")
        step_info = f"  шаг {meta.get('step', '?')}" if meta.get("step") else ""
        cost = f"  ${meta.get('cost', 0):.4f}" if meta.get("cost") else ""
        print(f"╰─ {meta.get('finish_reason', 'stop')}{step_info}{cost}")

    def status(self, state: CLIState) -> None:
        print(
            f"\n  [{state.working_state}] · {state.model} · ctx:{state.ctx_strategy}"
            f" · turns:{state.short_term_turns}"
        )

    def section(self, title: str) -> None:
        print(f"\n{self.SEP}\n  {title}\n{self.SEP}")

    def item(self, key: str, value: Any, active: bool = False) -> None:
        marker = " ●" if active else "  "
        value_str = str(value)
        print(f"{marker} {key:<28} {value_str}")

    def success(self, msg: str) -> None:
        print(f"\n  ✓ {msg}")

    def error(self, msg: str) -> None:
        print(f"\n  ✗ {msg}", file=sys.stderr)

    def info(self, msg: str) -> None:
        print(f"\n  → {msg}")

    def help(self) -> None:
        self.section("Команды")
        groups = [
            (
                "Общие",
                [
                    ("/help", "эта справка"),
                    ("/status", "текущее состояние сессии"),
                    ("/exit", "выйти"),
                ],
            ),
            (
                "Модель",
                [
                    ("/settings model", "показать модель и список доступных"),
                    ("/settings model set <id>", "переключить модель"),
                ],
            ),
            (
                "Задача",
                [
                    ("/settings task", "состояние задачи и память"),
                    ("/settings task ctx <стратегия>", "сменить стратегию контекста"),
                    ("/settings task intent <intent>", "отправить сообщение с явным client_intent"),
                    ("/settings task checkpoint <имя>", "создать checkpoint (только branching)"),
                    ("/settings task fork <checkpoint> [ветка]", "форк ветки (только branching)"),
                    ("/settings task branch <имя>", "переключить ветку (только branching)"),
                    ("/settings task memory", "показать слои памяти"),
                    (
                        "/settings task memory clear <layer>",
                        "очистить слой (short-term/working/long-term)",
                    ),
                    ("/settings task profile", "показать профиль"),
                    ("/settings task reset", "сбросить сессию"),
                ],
            ),
            (
                "Проекты",
                [
                    ("/settings project", "список проектов"),
                    ("/settings project new <имя>", "создать проект"),
                    ("/settings project use <id>", "активировать проект"),
                    ("/settings project delete <id>", "удалить проект"),
                ],
            ),
            (
                "MCP",
                [
                    ("/settings mcp", "список MCP-инструментов"),
                    ("/settings mcp call <tool> [k=v ...]", "вызвать инструмент MCP Time"),
                ],
            ),
        ]
        for group, cmds in groups:
            print(f"\n  {group}:")
            for cmd, desc in cmds:
                print(f"    {cmd:<44} {desc}")


# ── Диспетчер команд ─────────────────────────────────────────────────────────


class CommandRouter:
    """Разбирает строку команды и вызывает нужный обработчик."""

    CTX_STRATEGIES = ["sliding_window", "sticky_facts", "branching", "history_compression"]
    VALID_MEMORY_LAYERS = {"short-term", "working", "long-term"}
    VALID_INTENTS = {
        "task_intent",
        "plan_formation_intent",
        "decision_memory_write",
        "note_memory_write",
        "start_execution",
        "plan_approved",
        "plan_formation",
        "skip_mandatory_planning",
        "goal_clarification",
        "direct_code_request",
        "validation_request",
        "validation_checklist_request",
        "validation_confirm",
        "validation_reject",
        "validation_skip_request",
        "yes_confirmation",
        "no_confirmation",
        "stack_switch_request",
        "third_party_dependency_request",
        "step_completed",
        "working_update",
        "confirm_pending_memory",
    }

    def __init__(self, api: APIClient, state: CLIState, renderer: Renderer):
        self._api = api
        self._s = state
        self._r = renderer

    def dispatch(self, raw: str) -> bool:
        """Возвращает False если нужно выйти."""
        parts = raw.strip().split()
        if not parts:
            return True
        cmd, args = parts[0].lower(), parts[1:]

        if cmd == "/exit":
            return False
        if cmd == "/help":
            self._r.help()
            return True
        if cmd == "/status":
            self._cmd_status()
            return True
        if cmd == "/settings" and args:
            self._settings(args)
            return True
        if cmd == "/settings":
            self._r.help()
            return True

        self._r.error(f"Неизвестная команда '{cmd}'. Введите /help")
        return True

    def _cmd_status(self) -> None:
        self._r.section("Текущее состояние")
        self._r.item("Модель", self._s.model)
        self._r.item("Провайдер", self._s.provider)
        self._r.item("Стратегия ctx", self._s.ctx_strategy)
        self._r.item("Проект", self._s.project_name)
        self._r.item("Состояние задачи", self._s.working_state)
        self._r.item("Сообщений (st)", self._s.short_term_turns)

    def _settings(self, args: list[str]) -> None:
        group = args[0].lower()
        rest = args[1:]
        if group == "model":
            self._settings_model(rest)
        elif group == "task":
            self._settings_task(rest)
        elif group == "project":
            self._settings_project(rest)
        elif group == "mcp":
            self._settings_mcp(rest)
        else:
            self._r.error("Неизвестная группа настроек. Доступно: model, task, project, mcp")

    # ── /settings model ───────────────────────────────────────────────────────

    def _settings_model(self, args: list[str]) -> None:
        if not args:
            resp = self._api.get_models()
            if not resp.ok:
                self._r.error(f"Не удалось получить список моделей: {resp.error}")
                return
            self._r.section("Модели")
            current = str(resp.data.get("current_model") or "")
            models = resp.data.get("available_models") or []
            if not isinstance(models, list):
                models = []
            for model_name in models:
                active = str(model_name) == current
                self._r.item(str(model_name), "текущая" if active else "", active=active)
            return

        if args[0] == "set" and len(args) >= 2:
            model = args[1]
            resp = self._api.set_model(model)
            if resp.ok:
                current = str(resp.data.get("current_model") or model)
                self._s.model = current
                self._s.provider = _infer_provider(current)
                self._r.success(f"Модель переключена на {current}")
            else:
                self._r.error(f"Ошибка: {resp.error}")
            return

        self._r.error("Использование: /settings model  ИЛИ  /settings model set <model_id>")

    # ── /settings task ────────────────────────────────────────────────────────

    def _settings_task(self, args: list[str]) -> None:
        if not args:
            self._cmd_status()
            self._show_memory_layers()
            return

        sub = args[0].lower()
        rest = args[1:]

        if sub == "ctx":
            self._task_ctx(rest)
            return
        if sub == "memory":
            self._task_memory(rest)
            return
        if sub == "profile":
            self._task_profile()
            return
        if sub == "checkpoint":
            self._task_checkpoint(rest)
            return
        if sub == "fork":
            self._task_fork(rest)
            return
        if sub == "branch":
            self._task_branch(rest)
            return
        if sub == "intent":
            self._task_intent(rest)
            return
        if sub == "reset":
            self._task_reset()
            return

        self._r.error(f"Неизвестная подкоманда task '{sub}'. Введите /help")

    def _task_ctx(self, rest: list[str]) -> None:
        if not rest:
            self._r.section("Стратегии контекста")
            for strategy in self.CTX_STRATEGIES:
                self._r.item(strategy, "", active=(strategy == self._s.ctx_strategy))
            self._r.info("Использование: /settings task ctx <стратегия>")
            return

        strategy = rest[0]
        if strategy not in self.CTX_STRATEGIES:
            self._r.error(f"Неизвестная стратегия '{strategy}'. Доступно: {', '.join(self.CTX_STRATEGIES)}")
            return

        resp = self._api.set_ctx_strategy(strategy, self._s.session_id)
        if resp.ok:
            self._s.ctx_strategy = str(resp.data.get("strategy") or strategy)
            self._r.success(f"Стратегия контекста: {self._s.ctx_strategy}")
        else:
            self._r.error(f"Ошибка: {resp.error}")

    def _show_memory_layers(self) -> None:
        resp = self._api.get_memory_layers(self._s.session_id)
        if not resp.ok:
            self._r.error(f"Ошибка: {resp.error}")
            return

        short_term = resp.data.get("short_term") or {}
        working = resp.data.get("working") or {}
        long_term = resp.data.get("long_term") or {}
        decisions = long_term.get("decisions_top_k") or []
        notes = long_term.get("notes_top_k") or []

        turns_count = short_term.get("turns_count", "—")
        if isinstance(turns_count, int):
            self._s.short_term_turns = turns_count

        self._r.section("Слои памяти")
        self._r.item("short-term сообщений", turns_count)
        self._r.item("working layer", "есть" if working.get("present") else "пусто")
        self._r.item("long-term decisions", len(decisions) if isinstance(decisions, list) else 0)
        self._r.item("long-term notes", len(notes) if isinstance(notes, list) else 0)

    def _task_memory(self, rest: list[str]) -> None:
        if not rest:
            self._show_memory_layers()
            return

        if rest[0] == "clear" and len(rest) >= 2:
            layer = rest[1]
            if layer not in self.VALID_MEMORY_LAYERS:
                self._r.error(f"Слой '{layer}' не существует. Доступно: {', '.join(sorted(self.VALID_MEMORY_LAYERS))}")
                return
            resp = self._api.clear_memory(layer, self._s.session_id)
            if resp.ok:
                self._r.success(f"Слой {layer} очищен")
            else:
                self._r.error(f"Ошибка: {resp.error}")
            return

        self._r.error("Использование: /settings task memory  ИЛИ  /settings task memory clear <layer>")

    def _task_profile(self) -> None:
        resp = self._api.get_profile(self._s.session_id)
        if not resp.ok:
            self._r.error(f"Ошибка: {resp.error}")
            return

        profile = resp.data.get("profile", resp.data)
        self._r.section("Профиль пользователя")
        if isinstance(profile, dict):
            printed = 0
            for key, value in profile.items():
                if value in (None, "", [], {}, False):
                    continue
                self._r.item(str(key), self._compact_value(value))
                printed += 1
            if printed == 0:
                self._r.item("profile", "пусто")
        else:
            self._r.item("profile", self._compact_value(profile))

    def _task_checkpoint(self, rest: list[str]) -> None:
        if self._s.ctx_strategy != "branching":
            self._r.error(
                "checkpoint доступен только при стратегии branching "
                f"(сейчас: {self._s.ctx_strategy}). Переключите: /settings task ctx branching"
            )
            return

        name = " ".join(rest).strip() if rest else "checkpoint"
        resp = self._api.checkpoint(self._s.session_id, name)
        if resp.ok:
            checkpoint_info = resp.data.get("checkpoint") or {}
            checkpoint_name = checkpoint_info.get("name") if isinstance(checkpoint_info, dict) else None
            self._r.success(f"Checkpoint '{checkpoint_name or name}' создан")
        else:
            self._r.error(f"Ошибка: {resp.error}")

    def _task_fork(self, rest: list[str]) -> None:
        if self._s.ctx_strategy != "branching":
            self._r.error(
                "fork доступен только при стратегии branching "
                f"(сейчас: {self._s.ctx_strategy}). Переключите: /settings task ctx branching"
            )
            return
        if not rest:
            self._r.error("Использование: /settings task fork <checkpoint> [ветка]")
            return

        checkpoint = rest[0]
        branch_name = " ".join(rest[1:]).strip() if len(rest) > 1 else None
        resp = self._api.fork(self._s.session_id, checkpoint=checkpoint, branch_name=branch_name)
        if resp.ok:
            new_branch = str(resp.data.get("new_branch") or branch_name or "новая_ветка")
            self._r.success(f"Форк создан: checkpoint='{checkpoint}' -> branch='{new_branch}'")
        else:
            self._r.error(f"Ошибка: {resp.error}")

    def _task_branch(self, rest: list[str]) -> None:
        if not rest:
            self._r.error("Использование: /settings task branch <имя>")
            return
        if self._s.ctx_strategy != "branching":
            self._r.error(
                "switch-branch доступен только при стратегии branching "
                f"(сейчас: {self._s.ctx_strategy}). Переключите: /settings task ctx branching"
            )
            return
        resp = self._api.switch_branch(self._s.session_id, " ".join(rest).strip())
        if resp.ok:
            active = str(resp.data.get("active_branch") or " ".join(rest).strip())
            self._r.success(f"Ветка переключена на '{active}'")
        else:
            self._r.error(f"Ошибка: {resp.error}")

    def _task_intent(self, rest: list[str]) -> None:
        if not rest:
            self._r.section("Доступные client_intent (для CLI)")
            for intent_name in sorted(self.VALID_INTENTS):
                self._r.item(intent_name, "")
            self._r.info("Использование: /settings task intent <intent_name> [сообщение]")
            return

        intent_name = rest[0]
        if intent_name not in self.VALID_INTENTS:
            self._r.error(f"Неизвестный intent '{intent_name}'. Введите /settings task intent для списка")
            return

        message = " ".join(rest[1:]).strip() if len(rest) > 1 else intent_name
        resp = self._api.chat_with_intent(message, intent_name, self._s.session_id, self._s.project_id)
        if not resp.ok:
            self._r.error(f"Ошибка: {resp.error}")
            return

        reply = str(resp.data.get("reply") or "")
        _update_state_from_response(self._s, resp.data)
        self._r.assistant(reply, _extract_meta(self._s, resp.data))
        self._r.status(self._s)

    def _task_reset(self) -> None:
        resp = self._api.reset_session(self._s.session_id)
        if resp.ok:
            self._s.short_term_turns = 0
            self._s.working_state = "IDLE"
            self._r.success("Сессия сброшена")
        else:
            self._r.error(f"Ошибка: {resp.error}")

    @staticmethod
    def _compact_value(value: Any, limit: int = 70) -> str:
        if isinstance(value, (dict, list)):
            text = json.dumps(value, ensure_ascii=False)
        else:
            text = str(value)
        if len(text) <= limit:
            return text
        return text[: limit - 1] + "…"

    @staticmethod
    def _parse_cli_value(raw: str) -> Any:
        """Пытается распарсить JSON-литерал, иначе возвращает строку как есть."""
        value = str(raw or "")
        if value == "":
            return ""
        try:
            return json.loads(value)
        except Exception:
            return value

    def _parse_key_value_pairs(self, tokens: list[str]) -> tuple[dict[str, Any], str | None]:
        """Парсит аргументы вида key=value."""
        args: dict[str, Any] = {}
        for token in tokens:
            if "=" not in token:
                return {}, f"Аргумент '{token}' должен быть в формате key=value"
            key, raw_value = token.split("=", 1)
            name = key.strip()
            if not name:
                return {}, f"Пустой ключ в аргументе '{token}'"
            args[name] = self._parse_cli_value(raw_value.strip())
        return args, None

    # ── /settings project ─────────────────────────────────────────────────────

    def _settings_project(self, args: list[str]) -> None:
        if not args:
            resp = self._api.get_projects()
            if not resp.ok:
                self._r.error(f"Ошибка: {resp.error}")
                return
            self._r.section("Проекты")
            projects = resp.data.get("projects") or []
            for project in projects:
                pid = str(project.get("id") or "—")
                name = str(project.get("name") or "—")
                active = bool(project.get("is_active")) or pid == self._s.project_id
                self._r.item(name, pid, active=active)
            return

        sub = args[0].lower()
        rest = args[1:]

        if sub == "new":
            name = " ".join(rest).strip() if rest else "Новый проект"
            if len(name) > 50:
                self._r.error(f"Имя проекта не должно превышать 50 символов (сейчас {len(name)})")
                return

            resp = self._api.create_project(name)
            if resp.ok:
                self._s.project_id = str(resp.data.get("id") or self._s.project_id)
                self._s.session_id = str(resp.data.get("session_id") or self._s.session_id)
                self._s.project_name = str(resp.data.get("name") or name)
                self._s.working_state = "IDLE"
                self._s.short_term_turns = 0
                self._r.success(f"Проект '{self._s.project_name}' создан и активирован  id={self._s.project_id}")
            else:
                self._r.error(f"Ошибка: {resp.error}")
            return

        if sub == "use":
            if not rest:
                self._r.error("Использование: /settings project use <project_id>")
                return
            target_id = rest[0]
            resp = self._api.activate_project(target_id)
            if resp.ok:
                self._s.project_id = target_id
                self._s.session_id = str(resp.data.get("session_id") or self._s.session_id)
                self._refresh_current_project_name()
                self._s.working_state = "IDLE"
                self._r.success(f"Проект {target_id} активирован")
            else:
                self._r.error(f"Ошибка: {resp.error}")
            return

        if sub == "delete":
            if not rest:
                self._r.error("Использование: /settings project delete <project_id>")
                return
            target_id = rest[0]
            if target_id == self._s.project_id:
                self._r.error("Нельзя удалить активный проект. Сначала переключитесь: /settings project use <другой_id>")
                return
            resp = self._api.delete_project(target_id)
            if resp.ok:
                self._r.success(f"Проект {target_id} удалён")
            else:
                self._r.error(f"Ошибка: {resp.error}")
            return

        self._r.error(f"Неизвестная подкоманда project '{sub}'. Введите /help")

    def _refresh_current_project_name(self) -> None:
        resp = self._api.get_projects()
        if not resp.ok:
            return
        projects = resp.data.get("projects") or []
        for project in projects:
            pid = str(project.get("id") or "")
            if pid == self._s.project_id:
                self._s.project_name = str(project.get("name") or self._s.project_name)
                if project.get("session_id"):
                    self._s.session_id = str(project.get("session_id"))
                return

    # ── /settings mcp ─────────────────────────────────────────────────────────

    def _settings_mcp(self, args: list[str]) -> None:
        target = args[0].lower() if args else "all"

        if target == "all":
            resp = self._api.get_mcp_tools()
            if not resp.ok:
                self._r.error(f"MCP Time недоступен: {resp.error}")
                return
            self._r.section("MCP Time — инструменты")
            for tool in resp.data.get("tools", []):
                self._r.item(str(tool.get("name", "—")), str(tool.get("description", "")))
            return

        if target == "call":
            if len(args) < 2:
                self._r.error("Использование: /settings mcp call <tool> [key=value ...]")
                return
            tool_name = args[1]
            call_args, parse_error = self._parse_key_value_pairs(args[2:])
            if parse_error:
                self._r.error(parse_error)
                return

            resp = self._api.call_mcp_tool(tool_name, call_args)
            if not resp.ok:
                self._r.error(f"MCP call error: {resp.error}")
                return

            result = resp.data.get("result") or {}
            self._r.section(f"MCP Time — {tool_name}")
            if call_args:
                self._r.item("arguments", self._compact_value(call_args, limit=120))

            text_output = str(result.get("text") or "").strip()
            if text_output:
                for line in text_output.splitlines():
                    self._r.item("text", self._compact_value(line, limit=120))

            structured = result.get("structured")
            if structured not in (None, {}, []):
                self._r.item("structured", self._compact_value(structured, limit=120))

            self._r.success(f"Инструмент '{tool_name}' выполнен")
            return

        self._r.error(f"Неизвестная подкоманда mcp '{target}'. Доступно: call")


# ── Инициализация сессии ──────────────────────────────────────────────────────


def _init_session(api: APIClient, state: CLIState, renderer: Renderer) -> bool:
    """
    Bootstrap-паттерн строго по README:
    1. GET /session/restore → если found=true, взять session_id + project_id
    2. Если found=false + needs_project=true → POST /projects
    3. Только после restore/create → заполнить CLIState
    """
    resp_models = api.get_models()
    if not resp_models.ok:
        renderer.error(
            f"Сервер недоступен: {resp_models.error}\n"
            "  Убедитесь что сервер запущен: python3 app.py"
        )
        return False

    state.model = str(resp_models.data.get("current_model") or "—")
    state.provider = _infer_provider(state.model)

    resp_restore = api.restore_session()
    if not resp_restore.ok:
        renderer.error(f"Не удалось восстановить сессию: {resp_restore.error}")
        return False

    if resp_restore.data.get("found"):
        state.session_id = str(resp_restore.data.get("session_id") or "")
        state.project_id = str(resp_restore.data.get("project_id") or "")
        state.project_name = str(resp_restore.data.get("project_name") or "—")
        state.ctx_strategy = str(resp_restore.data.get("ctx_strategy") or "sticky_facts")
        state.model = str(resp_restore.data.get("current_model") or state.model)
        state.provider = _infer_provider(state.model)
        working_view = resp_restore.data.get("working_view") or {}
        if isinstance(working_view, dict):
            state.working_state = str(working_view.get("state") or "IDLE")
    else:
        if not resp_restore.data.get("needs_project", True):
            renderer.error("Сессия не найдена, но сервер не запросил создание проекта")
            return False
        resp_new = api.create_project("CLI Project")
        if not resp_new.ok:
            renderer.error(f"Не удалось создать проект: {resp_new.error}")
            return False
        state.project_id = str(resp_new.data.get("id") or "")
        state.session_id = str(resp_new.data.get("session_id") or "")
        state.project_name = str(resp_new.data.get("name") or "CLI Project")
        state.ctx_strategy = "sticky_facts"
        state.working_state = "IDLE"

    if not state.session_id or not state.project_id:
        renderer.error("Сервер вернул неполные данные сессии (session_id/project_id)")
        return False

    layers_resp = api.get_memory_layers(state.session_id)
    if layers_resp.ok:
        short_term = layers_resp.data.get("short_term") or {}
        turns = short_term.get("turns_count")
        if isinstance(turns, int):
            state.short_term_turns = turns

    return True


def _infer_provider(model: str) -> str:
    """Определяет провайдера по имени модели через префиксы."""
    model_lower = str(model or "").lower()
    if model_lower.startswith("gpt") or model_lower.startswith("o1") or model_lower.startswith("o3"):
        return "OpenAI"
    if model_lower.startswith("deepseek"):
        return "DeepSeek"
    if model_lower.startswith("claude"):
        return "Anthropic"
    return "Unknown"


def _update_state_from_response(state: CLIState, data: dict[str, Any]) -> None:
    """Обновляет CLIState из ответа /chat."""
    model = data.get("model")
    if model:
        state.model = str(model)
        state.provider = _infer_provider(state.model)

    wv = data.get("working_view", {})
    if isinstance(wv, dict):
        state.working_state = str(wv.get("state") or state.working_state)

    mem = data.get("memory_stats", {})
    if isinstance(mem, dict):
        turns = mem.get("short_term_messages")
        if isinstance(turns, int):
            state.short_term_turns = turns

    strategy = data.get("ctx_strategy")
    if strategy:
        state.ctx_strategy = str(strategy)


def _extract_meta(state: CLIState, data: dict[str, Any]) -> dict[str, Any]:
    """Собирает метаданные для Renderer.assistant() из ответа /chat."""
    wv = data.get("working_view", {})
    finish_reason = data.get("finish_reason") or data.get("token_stats", {}).get("finish_reason") or "stop"

    step_str = None
    if isinstance(wv, dict):
        step_index = wv.get("step_index")
        total_steps = wv.get("total_steps")
        if isinstance(step_index, int) and isinstance(total_steps, int) and total_steps > 0:
            step_str = f"{step_index}/{total_steps}"

    token_stats = data.get("token_stats") or {}
    cost = token_stats.get("cost_usd") if isinstance(token_stats, dict) else None

    return {
        "working_state": state.working_state,
        "finish_reason": finish_reason,
        "step": step_str,
        "cost": cost,
    }


# ── REPL-цикл ─────────────────────────────────────────────────────────────────


class REPLLoop:
    """Главный цикл ввода/вывода терминального клиента."""

    def __init__(self, api: APIClient, state: CLIState, renderer: Renderer, router: CommandRouter):
        self._api = api
        self._state = state
        self._renderer = renderer
        self._router = router

    def run(self) -> None:
        if not _init_session(self._api, self._state, self._renderer):
            sys.exit(1)

        self._renderer.header(self._state)

        while True:
            try:
                raw = input("\nВы: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n  До свидания.")
                break

            if not raw:
                continue

            if raw.startswith("/"):
                if not self._router.dispatch(raw):
                    print("  До свидания.")
                    break
                continue

            if not self._state.project_id:
                self._renderer.error("Нет активного проекта. Создайте: /settings project new <имя>")
                continue

            resp = self._api.chat(raw, self._state.session_id, self._state.project_id)
            if not resp.ok:
                self._renderer.error(f"Ошибка запроса: {resp.error}")
                continue

            reply = str(resp.data.get("reply") or "")
            _update_state_from_response(self._state, resp.data)
            self._renderer.assistant(reply, _extract_meta(self._state, resp.data))
            self._renderer.status(self._state)


def run() -> None:
    user_id = _load_or_create_user_id()
    api = APIClient(BASE_URL, user_id)
    state = CLIState()
    renderer = Renderer()
    router = CommandRouter(api, state, renderer)
    loop = REPLLoop(api, state, renderer, router)
    loop.run()


if __name__ == "__main__":
    run()
