import json
import logging
import os
import re
from uuid import uuid4

from flask import Flask, Response, jsonify, render_template, request

from agent import IOSAgent
from mcp_time import call_time_tool, get_time_tools
from storage import (
    clear_session_messages,
    count_projects,
    create_project,
    create_session,
    delete_project_with_session,
    get_active_project,
    get_project,
    get_project_by_session,
    init_db,
    activate_project as storage_activate_project,
    list_projects,
    load_session,
    save_ctx_state,
    save_message,
    session_exists,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("app")

app = Flask(__name__)
init_db()

agent = IOSAgent()
logger.info("[INIT] IOSAgent инициализирован")
_runtime_session_id: str | None = None

def _pretty_json(payload: dict) -> str:
    """Форматирует JSON для читаемых логов."""
    try:
        return json.dumps(payload, ensure_ascii=False, indent=2, default=str)
    except Exception:
        return str(payload)


def _compact_text_for_log(text: str, max_chars: int = 600) -> dict:
    """Подготавливает текст для логов: ограничивает длину и добавляет флаг обрезки."""
    value = str(text or "")
    normalized = re.sub(r"\s+", " ", value).strip()
    truncated = len(normalized) > max_chars
    preview = normalized[:max_chars] + ("…" if truncated else "")
    return {
        "preview": preview,
        "truncated": truncated,
        "chars": len(normalized),
    }


def _client_ip() -> str:
    """Возвращает IP клиента с учётом прокси-заголовка."""
    forwarded_for = request.headers.get("X-Forwarded-For", "")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    return request.remote_addr or "unknown"


def _log_http_request(route: str, payload: dict) -> None:
    """Логирует входящий HTTP-запрос."""
    log_payload = {
        "route": route,
        "method": request.method,
        "client_ip": _client_ip(),
        "payload": payload,
    }
    logger.info("[HTTP][Запрос]\n%s", _pretty_json(log_payload))


def _log_http_response(route: str, status_code: int, payload: dict) -> None:
    """Логирует исходящий HTTP-ответ."""
    log_payload = {
        "route": route,
        "status_code": status_code,
        "response": payload,
    }
    logger.info("[HTTP][Ответ]\n%s", _pretty_json(log_payload))


def _extract_context_overflow(error_text: str) -> dict | None:
    """Пытается извлечь фактический размер контекста из ошибки превышения лимита."""
    if not error_text:
        return None
    max_match = re.search(r"maximum context length is (\d+) tokens", error_text)
    used_match = re.search(r"messages resulted in (\d+) tokens", error_text)
    if not max_match and not used_match:
        return None
    return {
        "context_limit": int(max_match.group(1)) if max_match else 0,
        "prompt_tokens": int(used_match.group(1)) if used_match else 0,
    }


def _build_ctx_state() -> dict:
    """Формирует состояние активной context-стратегии для фронтенда."""
    stats = agent.ctx.stats(agent.conversation_history)
    return {
        "strategy": agent.ctx.active,
        "stats": stats,
        "memory": agent.last_memory_stats or {},
    }


def _extract_last_turn_for_storage(default_user: str, default_assistant: str) -> tuple[str, str]:
    """Возвращает последние user/assistant реплики для сохранения в БД."""
    if agent.ctx.active == "branching":
        branch_msgs = agent.ctx.strategy.branches.get(agent.ctx.strategy.active_branch, [])
        assistant_content = default_assistant
        user_content = default_user
        if branch_msgs:
            if branch_msgs[-1].get("role") == "assistant":
                assistant_content = branch_msgs[-1].get("content", default_assistant)
            for msg in reversed(branch_msgs):
                if msg.get("role") == "user":
                    user_content = msg.get("content", default_user)
                    break
        return user_content, assistant_content

    user_content = default_user
    if len(agent.conversation_history) >= 2:
        user_content = agent.conversation_history[-2].get("content", default_user)
    assistant_content = default_assistant
    if agent.conversation_history:
        assistant_content = agent.conversation_history[-1].get("content", default_assistant)
    return user_content, assistant_content


def _normalize_user_id(raw_value: str | None) -> str:
    return str(raw_value or "").strip()


def _get_or_create_user_id(explicit_user_id: str | None = None) -> str:
    """Читает user_id из запроса/cookie или создаёт новый."""
    requested = _normalize_user_id(explicit_user_id)
    if requested:
        return requested
    existing = _normalize_user_id(request.cookies.get("user_id"))
    if existing:
        return existing
    return f"user_{uuid4().hex}"


def _get_or_create_session(user_id: str | None = None, requested_session_id: str | None = None) -> str:
    """Возвращает session_id из запроса, активного проекта или создаёт fallback-сессию."""
    candidate = str(requested_session_id or "").strip()
    uid = _normalize_user_id(user_id)

    if candidate and session_exists(candidate):
        if not uid:
            return candidate
        project = get_project_by_session(user_id=uid, session_id=candidate)
        if project:
            return candidate

    cookie_sid = str(request.cookies.get("session_id") or "").strip()
    if cookie_sid and session_exists(cookie_sid):
        if not uid:
            return cookie_sid
        cookie_project = get_project_by_session(user_id=uid, session_id=cookie_sid)
        if cookie_project:
            return cookie_sid

    if uid:
        active_project = get_active_project(uid)
        if active_project:
            active_sid = str(active_project.get("session_id") or "").strip()
            if active_sid and session_exists(active_sid):
                return active_sid

    return create_session()


def _set_session_cookie(resp: Response, session_id: str | None = None, user_id: str | None = None) -> Response:
    """Устанавливает cookie сессии и пользователя на 30 дней."""
    if session_id:
        resp.set_cookie("session_id", session_id, max_age=60 * 60 * 24 * 30, httponly=True)
    if user_id:
        resp.set_cookie("user_id", user_id, max_age=60 * 60 * 24 * 30, httponly=True)
    return resp


def _project_public_payload(project: dict) -> dict:
    return {
        "id": project.get("id"),
        "name": project.get("name"),
        "created_at": project.get("created_at"),
        "session_id": project.get("session_id"),
        "is_active": bool(project.get("is_active")),
    }


def _resolve_request_user_id(payload: dict | None = None) -> str:
    raw_query = _normalize_user_id(request.args.get("user_id"))
    if raw_query:
        return _get_or_create_user_id(raw_query)
    body = payload or {}
    raw_body = _normalize_user_id(body.get("user_id"))
    return _get_or_create_user_id(raw_body)


def _ensure_active_project(user_id: str) -> dict | None:
    project = get_active_project(user_id)
    if not project:
        return None
    sid = str(project.get("session_id") or "").strip()
    if sid and session_exists(sid):
        return project
    return None


def _load_agent_runtime_session(session_id: str, user_id: str) -> dict | None:
    """
    Подгружает историю/ctx для нужной session_id, чтобы избежать смешивания проектов.
    """
    global _runtime_session_id
    if _runtime_session_id == session_id:
        return load_session(session_id)

    data = load_session(session_id)
    full_messages = data.get("messages", []) if data else []
    agent.conversation_history = [{"role": msg["role"], "content": msg["content"]} for msg in full_messages]
    agent.restore_memory_session(session_id=session_id, messages=agent.conversation_history)
    agent.last_memory_stats = agent.memory.stats(session_id=session_id, user_id=user_id)

    if data and data.get("ctx_state"):
        agent.ctx.restore(data["ctx_state"])
    else:
        agent.ctx.reset_all()

    _runtime_session_id = session_id
    return data


def _resolve_debug_memory_request(data: dict | None = None) -> tuple[str, str, str, int]:
    """Нормализует общие параметры debug memory запросов."""
    payload = data or {}
    user_id = _resolve_request_user_id(payload)
    requested_session_id = str(payload.get("session_id") or request.args.get("session_id") or "").strip()
    session_id = _get_or_create_session(user_id=user_id, requested_session_id=requested_session_id)
    query = str(payload.get("q") or request.args.get("q") or "").strip()
    try:
        top_k = int(payload.get("top_k", request.args.get("top_k", 3)))
    except (TypeError, ValueError):
        top_k = 3
    top_k = max(1, min(10, top_k))
    return user_id, session_id, query, top_k


def _build_debug_memory_snapshot(*, session_id: str, user_id: str, query: str, top_k: int) -> dict:
    snapshot = agent.memory.debug_snapshot(session_id=session_id, user_id=user_id, query=query, top_k=top_k)
    snapshot["resolved_session_id"] = session_id
    return snapshot


def _debug_memory_clear_response(route: str, *, session_id: str, user_id: str, query: str, top_k: int, cleared: bool) -> Response:
    snapshot = _build_debug_memory_snapshot(session_id=session_id, user_id=user_id, query=query, top_k=top_k)
    payload = {"success": True, "cleared": bool(cleared), "snapshot": snapshot}
    _log_http_response(
        route,
        200,
        {
            "success": True,
            "cleared": bool(cleared),
            "resolved_session_id": session_id,
            "short_term_turns": snapshot.get("short_term", {}).get("turns_count", 0),
            "working_present": snapshot.get("working", {}).get("present", False),
            "long_term_decisions": len(snapshot.get("long_term", {}).get("decisions_top_k") or []),
            "long_term_notes": len(snapshot.get("long_term", {}).get("notes_top_k") or []),
            "memory_writes": len(snapshot.get("memory_writes") or []),
        },
    )
    return jsonify(payload)


def _clear_runtime_conversation_context(*, session_id: str, user_id: str) -> None:
    """Сбрасывает runtime-историю и ctx для активной session_id без затрагивания working/long-term."""
    global _runtime_session_id
    if _runtime_session_id != session_id:
        return
    agent.conversation_history = []
    agent.ctx.reset_all()
    agent.restore_memory_session(session_id=session_id, messages=[])
    agent.last_memory_stats = agent.memory.stats(session_id=session_id, user_id=user_id)


@app.route("/")
def index():
    logger.info("[HTTP] Запрос GET /")
    user_id = _get_or_create_user_id()
    active_project = _ensure_active_project(user_id)
    session_id = str(active_project.get("session_id") or "") if active_project else None
    response = app.make_response(render_template("index.html"))
    return _set_session_cookie(response, session_id, user_id=user_id)


@app.route("/models", methods=["GET"])
def get_models():
    """Возвращает текущую модель и список доступных моделей."""
    payload = {
        "current_model": agent.model,
        "available_models": agent.available_models(),
    }
    _log_http_response("/models", 200, payload)
    return jsonify(payload)


@app.route("/mcp/time/tools", methods=["GET"])
def route_mcp_time_tools():
    """
    GET /mcp/time/tools
    Возвращает список инструментов MCP Time сервера.
    """
    result = get_time_tools()

    if not result.success:
        logger.error("[MCP_TIME] endpoint error: %s", result.error)
        return jsonify(
            {
                "success": False,
                "error": result.error,
                "tools": [],
            }
        ), 500

    logger.info("[MCP_TIME] endpoint ok, tools=%s", [tool.name for tool in result.tools])

    return jsonify(
        {
            "success": True,
            "count": len(result.tools),
            "tools": [
                {
                    "name": tool.name,
                    "description": tool.description,
                }
                for tool in result.tools
            ],
        }
    )


@app.route("/mcp/time/call", methods=["POST"])
def route_mcp_time_call():
    """
    POST /mcp/time/call
    Вызывает инструмент MCP Time по имени с аргументами.
    """
    data = request.get_json(silent=True) or {}
    tool = str(data.get("tool") or data.get("name") or "").strip()
    arguments = data.get("arguments")
    if not isinstance(arguments, dict):
        arguments = {}

    _log_http_request(
        "/mcp/time/call",
        {
            "tool": tool,
            "argument_keys": sorted(arguments.keys()),
        },
    )

    if not tool:
        payload = {"success": False, "error": "tool is required"}
        _log_http_response("/mcp/time/call", 400, payload)
        return jsonify(payload), 400

    result = call_time_tool(tool, arguments)
    if not result.success:
        payload = {
            "success": False,
            "tool": tool,
            "error": result.error or "MCP call failed",
        }
        _log_http_response("/mcp/time/call", 500, payload)
        return jsonify(payload), 500

    text_parts: list[str] = []
    for item in result.content:
        if isinstance(item, dict) and str(item.get("type") or "") == "text":
            text = str(item.get("text") or "").strip()
            if text:
                text_parts.append(text)
    combined_text = "\n".join(text_parts).strip()

    payload = {
        "success": not result.is_error,
        "tool": tool,
        "arguments": result.arguments,
        "result": {
            "is_error": result.is_error,
            "content": result.content,
            "structured": result.structured_content,
            "text": combined_text,
        },
    }

    if result.is_error:
        payload["error"] = combined_text or "tool returned error"
        _log_http_response(
            "/mcp/time/call",
            400,
            {"success": False, "tool": tool, "error": payload["error"]},
        )
        return jsonify(payload), 400

    _log_http_response(
        "/mcp/time/call",
        200,
        {
            "success": True,
            "tool": tool,
            "text_preview": _compact_text_for_log(combined_text),
        },
    )
    return jsonify(payload)


@app.route("/model", methods=["POST"])
def set_model():
    """Переключает модель для последующих запросов."""
    data = request.get_json(silent=True) or {}
    model = str(data.get("model", "")).strip()
    _log_http_request("/model", {"model": model})
    try:
        current_model = agent.set_model(model)
        payload = {
            "success": True,
            "current_model": current_model,
            "available_models": agent.available_models(),
        }
        _log_http_response("/model", 200, payload)
        return jsonify(payload)
    except ValueError as exc:
        payload = {
            "success": False,
            "error": str(exc),
            "current_model": agent.model,
            "available_models": agent.available_models(),
        }
        _log_http_response("/model", 400, payload)
        return jsonify(payload), 400


@app.route("/projects", methods=["GET"])
def get_projects():
    """Возвращает список проектов пользователя."""
    user_id = _resolve_request_user_id()
    _log_http_request("/projects", {"user_id": user_id[:12] + "…", "method": "GET"})

    _ensure_active_project(user_id)
    projects_payload = [_project_public_payload(project) for project in list_projects(user_id)]
    active_session = None
    for project in projects_payload:
        if project.get("is_active"):
            active_session = project.get("session_id")
            break
    _log_http_response("/projects", 200, {"count": len(projects_payload)})
    response = jsonify(projects_payload)
    return _set_session_cookie(response, str(active_session or ""), user_id=user_id)


@app.route("/projects", methods=["POST"])
def create_project_endpoint():
    """Создаёт новый проект и активирует его."""
    data = request.get_json(silent=True) or {}
    user_id = _resolve_request_user_id(data)
    name = str(data.get("name") or "").strip()
    _log_http_request("/projects", {"user_id": user_id[:12] + "…", "name": name, "method": "POST"})

    if not name:
        payload = {"success": False, "error": "name is required"}
        _log_http_response("/projects", 400, payload)
        return jsonify(payload), 400
    if len(name) > 50:
        payload = {"success": False, "error": "name must be at most 50 characters"}
        _log_http_response("/projects", 400, payload)
        return jsonify(payload), 400

    try:
        project = create_project(user_id=user_id, name=name, activate=True)
    except ValueError as exc:
        payload = {"success": False, "error": str(exc)}
        _log_http_response("/projects", 400, payload)
        return jsonify(payload), 400

    global _runtime_session_id
    _runtime_session_id = None
    payload = {
        "id": project["id"],
        "name": project["name"],
        "session_id": project["session_id"],
    }
    _log_http_response("/projects", 200, {"success": True, "id": project["id"], "session_id": project["session_id"]})
    response = jsonify(payload)
    return _set_session_cookie(response, project["session_id"], user_id=user_id)


@app.route("/projects/<project_id>/activate", methods=["PATCH"])
def activate_project_endpoint(project_id: str):
    """Активирует выбранный проект пользователя."""
    data = request.get_json(silent=True) or {}
    user_id = _resolve_request_user_id(data)
    pid = str(project_id or "").strip()
    _log_http_request("/projects/activate", {"user_id": user_id[:12] + "…", "project_id": pid})

    if not pid:
        payload = {"success": False, "error": "project_id is required"}
        _log_http_response("/projects/activate", 400, payload)
        return jsonify(payload), 400

    project = storage_activate_project(user_id=user_id, project_id=pid)
    if not project:
        payload = {"success": False, "error": "project not found"}
        _log_http_response("/projects/activate", 404, payload)
        return jsonify(payload), 404

    global _runtime_session_id
    _runtime_session_id = None
    payload = {"session_id": project["session_id"]}
    _log_http_response("/projects/activate", 200, {"success": True, "session_id": project["session_id"]})
    response = jsonify(payload)
    return _set_session_cookie(response, project["session_id"], user_id=user_id)


@app.route("/projects/<project_id>", methods=["DELETE"])
def delete_project_endpoint(project_id: str):
    """Удаляет проект пользователя вместе с его сессией."""
    data = request.get_json(silent=True) or {}
    user_id = _resolve_request_user_id(data)
    pid = str(project_id or "").strip()
    _log_http_request("/projects/delete", {"user_id": user_id[:12] + "…", "project_id": pid})

    if not pid:
        payload = {"success": False, "error": "project_id is required"}
        _log_http_response("/projects/delete", 400, payload)
        return jsonify(payload), 400

    project = get_project(project_id=pid, user_id=user_id)
    if not project:
        payload = {"success": False, "error": "project not found"}
        _log_http_response("/projects/delete", 404, payload)
        return jsonify(payload), 404

    if bool(project.get("is_active")):
        payload = {"success": False, "error": "Нельзя удалить активный проект. Сначала переключитесь на другой."}
        _log_http_response("/projects/delete", 409, payload)
        return jsonify(payload), 409

    if count_projects(user_id) <= 1:
        payload = {"success": False, "error": "Нельзя удалить последний проект."}
        _log_http_response("/projects/delete", 409, payload)
        return jsonify(payload), 409

    removed = delete_project_with_session(user_id=user_id, project_id=pid, delete_file=True)
    if not removed:
        payload = {"success": False, "error": "project not found"}
        _log_http_response("/projects/delete", 404, payload)
        return jsonify(payload), 404

    try:
        removed_sid = str(removed.get("session_id") or "").strip()
        if removed_sid:
            agent.clear_session_memory(removed_sid)
    except Exception:
        logger.exception("[PROJECTS] Не удалось очистить runtime memory для удалённой сессии")

    next_active = _ensure_active_project(user_id)
    next_session = str(next_active.get("session_id") or "") if next_active else ""
    global _runtime_session_id
    _runtime_session_id = None
    payload = {"success": True, "deleted": pid, "session_id": next_session}
    _log_http_response("/projects/delete", 200, payload)
    response = jsonify(payload)
    return _set_session_cookie(response, next_session, user_id=user_id)


@app.route("/chat", methods=["POST"])
def chat():
    """Принимает сообщение пользователя и возвращает ответ агента."""
    data = request.get_json(silent=True) or {}
    message = str(data.get("message", "")).strip()
    client_intent = data.get("client_intent")
    if not isinstance(client_intent, dict):
        client_intent = None
    requested_model = str(data.get("model", "")).strip()
    requested_session_id = str(data.get("session_id") or "").strip()
    user_id = _resolve_request_user_id(data)

    active_project = _ensure_active_project(user_id)

    if not active_project:
        payload = {"error": "Сначала создайте и выберите проект"}
        _log_http_response("/chat", 400, payload)
        return _set_session_cookie(jsonify(payload), None, user_id=user_id), 400

    session_id = str(active_project.get("session_id") or "").strip()
    if requested_session_id:
        requested_project = get_project_by_session(user_id=user_id, session_id=requested_session_id)
        if requested_project:
            if not bool(requested_project.get("is_active")):
                storage_activate_project(user_id=user_id, project_id=str(requested_project.get("id") or ""))
                active_project = _ensure_active_project(user_id) or requested_project
            else:
                active_project = requested_project
            session_id = requested_session_id

    if requested_model:
        try:
            agent.set_model(requested_model)
        except ValueError as exc:
            response_payload = {"error": str(exc), "current_model": agent.model}
            _log_http_response("/chat", 400, response_payload)
            return jsonify(response_payload), 400

    _log_http_request(
        "/chat",
        {
            "message": message,
            "client_intent": client_intent or None,
            "session_id": session_id,
            "user_id": user_id,
            "project_id": active_project.get("id"),
            "requested_model": requested_model or None,
            "current_model": agent.model,
        },
    )
    if not message:
        response_payload = {"error": "Пустое сообщение"}
        _log_http_response("/chat", 400, response_payload)
        return jsonify(response_payload), 400

    try:
        _load_agent_runtime_session(session_id=session_id, user_id=user_id)
        reply = agent.chat(
            message,
            session_id=session_id,
            user_id=user_id,
            client_intent=client_intent,
        )
        user_content, assistant_content = _extract_last_turn_for_storage(message, reply)
        save_message(session_id, "user", user_content)
        token_stats = agent.last_token_stats or {}
        save_message(
            session_id,
            "assistant",
            assistant_content,
            tokens_in=int(token_stats.get("prompt_tokens", 0) or 0),
            tokens_out=int(token_stats.get("completion_tokens", 0) or 0),
            cost_usd=float(token_stats.get("cost_usd", 0.0) or 0.0),
        )

        save_ctx_state(session_id, agent.ctx.dump())

        ctx_state = _build_ctx_state()
        working_actions = agent.memory.get_working_actions(session_id=session_id)
        response_meta = agent.last_chat_response_meta or {}
        working_view = response_meta.get("working_view") or agent.memory.get_working_view(session_id=session_id)
        finish_reason = response_meta.get("finish_reason")
        if finish_reason is None and isinstance(token_stats, dict):
            finish_reason = token_stats.get("finish_reason")
        response_payload = {
            "reply": reply,
            "model": agent.model,
            "token_stats": token_stats,
            "memory_stats": agent.last_memory_stats or {},
            "prompt_preview": agent.last_prompt_preview or {},
            "ctx_state": ctx_state,
            "ctx_stats": ctx_state["stats"],
            "ctx_strategy": ctx_state["strategy"],
            "working_view": working_view,
            "working_actions": response_meta.get("working_actions") or working_actions,
            "invariant_report": response_meta.get("invariant_report"),
            "internal": {
                "invariant_report": response_meta.get("invariant_report"),
                "internal_trace": response_meta.get("internal_trace"),
            },
            "finish_reason": finish_reason,
        }
        response = jsonify(response_payload)
        _log_http_response(
            "/chat",
            200,
            {
                "session_id": session_id,
                "reply_len": len(reply),
                "reply_preview": _compact_text_for_log(reply),
                "token_stats": token_stats,
                "memory_stats": agent.last_memory_stats or {},
            },
        )
        return _set_session_cookie(response, session_id, user_id=user_id)
    except Exception as exc:
        logger.exception("[CHAT] Ошибка обработки запроса чата: %s", exc)
        err_text = str(exc)
        overflow = _extract_context_overflow(err_text)
        response_payload = {"error": err_text}
        status = 500
        if overflow:
            response_payload["token_stats"] = {
                "prompt_tokens": int(overflow.get("prompt_tokens") or 0),
                "completion_tokens": 0,
                "total_tokens": int(overflow.get("prompt_tokens") or 0),
                "cost_usd": 0.0,
                "latency_ms": 0,
                "scope": "chat",
                "error": True,
                "context_limit": int(overflow.get("context_limit") or 0),
            }
            status = 400
        _log_http_response("/chat", status, response_payload)
        return jsonify(response_payload), status


@app.route("/debug/memory-layers", methods=["GET"])
def debug_memory_layers():
    """Возвращает снимок трёх слоёв памяти для Debug UI."""
    user_id = _resolve_request_user_id()
    requested_session_id = str(request.args.get("session_id") or "").strip()
    session_id = _get_or_create_session(user_id=user_id, requested_session_id=requested_session_id)
    query = (request.args.get("q") or "").strip()
    try:
        top_k = int(request.args.get("top_k", 3))
    except (TypeError, ValueError):
        top_k = 3
    top_k = max(1, min(10, top_k))
    _log_http_request(
        "/debug/memory-layers",
        {"session_id": session_id[:8] + "…", "user_id": user_id[:12] + "…", "q": query[:50] or "(empty)", "top_k": top_k},
    )
    try:
        snapshot = agent.memory.debug_snapshot(
            session_id=session_id,
            user_id=user_id,
            query=query,
            top_k=top_k,
        )
        snapshot["resolved_session_id"] = session_id
        summary = {
            "short_term_turns": snapshot.get("short_term", {}).get("turns_count", 0),
            "working_present": snapshot.get("working", {}).get("present", False),
            "long_term_decisions": len(snapshot.get("long_term", {}).get("decisions_top_k") or []),
            "long_term_notes": len(snapshot.get("long_term", {}).get("notes_top_k") or []),
            "memory_writes": len(snapshot.get("memory_writes") or []),
        }
        _log_http_response("/debug/memory-layers", 200, summary)
        return jsonify(snapshot)
    except Exception as exc:
        logger.exception("[DEBUG] memory-layers failed: %s", exc)
        _log_http_response("/debug/memory-layers", 500, {"error": str(exc)})
        return jsonify({"error": str(exc)}), 500


@app.route("/debug/memory/profile", methods=["GET"])
def debug_get_memory_profile():
    """Возвращает полный профиль long-term памяти с метаданными полей и конфликтами."""
    user_id = _resolve_request_user_id()
    requested_session_id = str(request.args.get("session_id") or "").strip()
    session_id = _get_or_create_session(user_id=user_id, requested_session_id=requested_session_id)
    _log_http_request(
        "/debug/memory/profile",
        {"session_id": session_id[:8] + "…", "user_id": user_id[:12] + "…"},
    )
    try:
        profile = agent.memory.get_profile_snapshot(user_id=user_id, session_id=session_id)
        payload = {
            "success": True,
            "profile": profile,
            "memory_writes": agent.memory.get_recent_write_events(session_id=session_id, limit=10),
        }
        if isinstance(profile, dict):
            # Backward-compatible shape for legacy scripts expecting top-level profile fields.
            payload.update(profile)
        _log_http_response(
            "/debug/memory/profile",
            200,
            {"success": True, "fields": list((profile or {}).keys())},
        )
        return jsonify(payload)
    except Exception as exc:
        logger.exception("[DEBUG] memory profile get failed: %s", exc)
        _log_http_response("/debug/memory/profile", 500, {"error": str(exc)})
        return jsonify({"success": False, "error": str(exc)}), 500

@app.route("/debug/memory/profile/field", methods=["PATCH"])
def debug_patch_memory_profile_field():
    """Обновляет поле профиля long-term памяти через DebugMenu."""
    data = request.get_json(silent=True) or {}
    user_id = _resolve_request_user_id(data)
    requested_session_id = str(data.get("session_id") or "").strip()
    session_id = _get_or_create_session(user_id=user_id, requested_session_id=requested_session_id)
    field = str(data.get("field") or "").strip()
    value = data.get("value")
    _log_http_request(
        "/debug/memory/profile/field",
        {"session_id": session_id[:8] + "…", "user_id": user_id[:12] + "…", "field": field},
    )
    if not field:
        payload = {"success": False, "error": "field is required"}
        _log_http_response("/debug/memory/profile/field", 400, payload)
        return jsonify(payload), 400
    try:
        profile = agent.memory.debug_update_profile_field(
            session_id=session_id,
            user_id=user_id,
            field=field,
            value=value,
        )
        payload = {
            "success": True,
            "profile": profile,
            "memory_writes": agent.memory.get_recent_write_events(session_id=session_id, limit=10),
        }
        _log_http_response("/debug/memory/profile/field", 200, {"success": True, "field": field, "operation": "patch"})
        return jsonify(payload)
    except Exception as exc:
        _log_http_response("/debug/memory/profile/field", 400, {"error": str(exc)})
        return jsonify({"success": False, "error": str(exc)}), 400


@app.route("/debug/memory/profile/field", methods=["DELETE"])
def debug_delete_memory_profile_field():
    """Удаляет поле профиля long-term памяти (каноническое или extra)."""
    data = request.get_json(silent=True) or {}
    user_id = _resolve_request_user_id(data)
    requested_session_id = str(data.get("session_id") or "").strip()
    session_id = _get_or_create_session(user_id=user_id, requested_session_id=requested_session_id)
    field = str(data.get("field") or "").strip()
    _log_http_request(
        "/debug/memory/profile/field",
        {"session_id": session_id[:8] + "…", "user_id": user_id[:12] + "…", "field": field, "method": "DELETE"},
    )
    if not field:
        payload = {"success": False, "error": "field is required"}
        _log_http_response("/debug/memory/profile/field", 400, payload)
        return jsonify(payload), 400
    try:
        profile = agent.memory.debug_delete_profile_field(
            session_id=session_id,
            user_id=user_id,
            field=field,
        )
        payload = {
            "success": True,
            "profile": profile,
            "memory_writes": agent.memory.get_recent_write_events(session_id=session_id, limit=10),
        }
        _log_http_response("/debug/memory/profile/field", 200, {"success": True, "field": field, "operation": "delete"})
        return jsonify(payload)
    except Exception as exc:
        _log_http_response("/debug/memory/profile/field", 400, {"error": str(exc)})
        return jsonify({"success": False, "error": str(exc)}), 400


@app.route("/debug/memory/profile/field", methods=["POST"])
def debug_add_memory_profile_field():
    """Добавляет новое поле в extra_fields профиля long-term памяти."""
    data = request.get_json(silent=True) or {}
    user_id = _resolve_request_user_id(data)
    requested_session_id = str(data.get("session_id") or "").strip()
    session_id = _get_or_create_session(user_id=user_id, requested_session_id=requested_session_id)
    field = str(data.get("field") or "").strip()
    value = data.get("value")
    _log_http_request(
        "/debug/memory/profile/field",
        {"session_id": session_id[:8] + "…", "user_id": user_id[:12] + "…", "field": field, "method": "POST"},
    )
    if not field:
        payload = {"success": False, "error": "field is required"}
        _log_http_response("/debug/memory/profile/field", 400, payload)
        return jsonify(payload), 400
    try:
        profile = agent.memory.debug_add_profile_extra_field(
            session_id=session_id,
            user_id=user_id,
            field=field,
            value=value,
        )
        payload = {
            "success": True,
            "profile": profile,
            "memory_writes": agent.memory.get_recent_write_events(session_id=session_id, limit=10),
        }
        _log_http_response("/debug/memory/profile/field", 200, {"success": True, "field": field, "operation": "add_extra"})
        return jsonify(payload)
    except Exception as exc:
        _log_http_response("/debug/memory/profile/field", 400, {"error": str(exc)})
        return jsonify({"success": False, "error": str(exc)}), 400


@app.route("/debug/memory/profile/confirm", methods=["POST"])
def debug_confirm_memory_profile_field():
    """Подтверждает поле профиля (verified=true)."""
    data = request.get_json(silent=True) or {}
    user_id = _resolve_request_user_id(data)
    requested_session_id = str(data.get("session_id") or "").strip()
    session_id = _get_or_create_session(user_id=user_id, requested_session_id=requested_session_id)
    field = str(data.get("field") or "").strip()
    _log_http_request(
        "/debug/memory/profile/confirm",
        {"session_id": session_id[:8] + "…", "user_id": user_id[:12] + "…", "field": field},
    )
    if not field:
        payload = {"success": False, "error": "field is required"}
        _log_http_response("/debug/memory/profile/confirm", 400, payload)
        return jsonify(payload), 400
    try:
        profile = agent.memory.debug_confirm_profile_field(
            session_id=session_id,
            user_id=user_id,
            field=field,
        )
        payload = {
            "success": True,
            "profile": profile,
            "memory_writes": agent.memory.get_recent_write_events(session_id=session_id, limit=10),
        }
        _log_http_response("/debug/memory/profile/confirm", 200, {"success": True, "field": field})
        return jsonify(payload)
    except Exception as exc:
        _log_http_response("/debug/memory/profile/confirm", 400, {"error": str(exc)})
        return jsonify({"success": False, "error": str(exc)}), 400


@app.route("/debug/memory/profile/conflict/resolve", methods=["POST"])
def debug_resolve_memory_profile_conflict():
    """Разрешает конфликт inferred-значения профиля."""
    data = request.get_json(silent=True) or {}
    user_id = _resolve_request_user_id(data)
    requested_session_id = str(data.get("session_id") or "").strip()
    session_id = _get_or_create_session(user_id=user_id, requested_session_id=requested_session_id)
    field = str(data.get("field") or "").strip()
    chosen_value = data.get("chosen_value")
    keep_existing = bool(data.get("keep_existing", False))
    _log_http_request(
        "/debug/memory/profile/conflict/resolve",
        {
            "session_id": session_id[:8] + "…",
            "user_id": user_id[:12] + "…",
            "field": field,
            "keep_existing": keep_existing,
            "has_chosen_value": chosen_value is not None,
        },
    )
    if not field:
        payload = {"success": False, "error": "field is required"}
        _log_http_response("/debug/memory/profile/conflict/resolve", 400, payload)
        return jsonify(payload), 400
    try:
        profile = agent.memory.debug_resolve_profile_conflict(
            session_id=session_id,
            user_id=user_id,
            field=field,
            chosen_value=chosen_value,
            keep_existing=keep_existing,
        )
        payload = {
            "success": True,
            "profile": profile,
            "memory_writes": agent.memory.get_recent_write_events(session_id=session_id, limit=10),
        }
        _log_http_response("/debug/memory/profile/conflict/resolve", 200, {"success": True, "field": field})
        return jsonify(payload)
    except Exception as exc:
        _log_http_response("/debug/memory/profile/conflict/resolve", 400, {"error": str(exc)})
        return jsonify({"success": False, "error": str(exc)}), 400


@app.route("/debug/memory/short-term/clear", methods=["POST"])
def debug_clear_short_term_memory():
    """Очищает short-term и историю сообщений текущей сессии для полного forget-контекста."""
    data = request.get_json(silent=True) or {}
    user_id, session_id, query, top_k = _resolve_debug_memory_request(data)
    _log_http_request(
        "/debug/memory/short-term/clear",
        {"session_id": session_id[:8] + "…", "user_id": user_id[:12] + "…", "top_k": top_k},
    )
    try:
        cleared = agent.memory.clear_short_term_layer(session_id=session_id)
        _clear_runtime_conversation_context(session_id=session_id, user_id=user_id)
        return _debug_memory_clear_response(
            "/debug/memory/short-term/clear",
            session_id=session_id,
            user_id=user_id,
            query=query,
            top_k=top_k,
            cleared=cleared,
        )
    except Exception as exc:
        logger.exception("[DEBUG] memory short-term clear failed: %s", exc)
        _log_http_response("/debug/memory/short-term/clear", 500, {"error": str(exc)})
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/debug/memory/working/clear", methods=["POST"])
def debug_clear_working_memory():
    """Очищает только рабочую память текущей сессии."""
    data = request.get_json(silent=True) or {}
    user_id, session_id, query, top_k = _resolve_debug_memory_request(data)
    _log_http_request(
        "/debug/memory/working/clear",
        {"session_id": session_id[:8] + "…", "user_id": user_id[:12] + "…", "top_k": top_k},
    )
    try:
        cleared = agent.memory.clear_working_layer(session_id=session_id)
        return _debug_memory_clear_response(
            "/debug/memory/working/clear",
            session_id=session_id,
            user_id=user_id,
            query=query,
            top_k=top_k,
            cleared=cleared,
        )
    except Exception as exc:
        logger.exception("[DEBUG] memory working clear failed: %s", exc)
        _log_http_response("/debug/memory/working/clear", 500, {"error": str(exc)})
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/debug/memory/long-term/clear", methods=["POST"])
def debug_clear_long_term_memory():
    """Очищает весь long-term слой пользователя."""
    data = request.get_json(silent=True) or {}
    user_id, session_id, query, top_k = _resolve_debug_memory_request(data)
    _log_http_request(
        "/debug/memory/long-term/clear",
        {"session_id": session_id[:8] + "…", "user_id": user_id[:12] + "…", "top_k": top_k},
    )
    try:
        cleared = agent.memory.clear_long_term_layer(session_id=session_id, user_id=user_id)
        return _debug_memory_clear_response(
            "/debug/memory/long-term/clear",
            session_id=session_id,
            user_id=user_id,
            query=query,
            top_k=top_k,
            cleared=cleared,
        )
    except Exception as exc:
        logger.exception("[DEBUG] memory long-term clear failed: %s", exc)
        _log_http_response("/debug/memory/long-term/clear", 500, {"error": str(exc)})
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/debug/memory/long-term/delete", methods=["POST"])
def debug_delete_longterm_entry():
    """Удаляет конкретную запись из long-term памяти (decision/note)."""
    data = request.get_json(silent=True) or {}
    user_id, session_id, query, top_k = _resolve_debug_memory_request(data)
    entry_type = str(data.get("entry_type") or "").strip().lower()
    entry_id_raw = data.get("id")
    _log_http_request(
        "/debug/memory/long-term/delete",
        {
            "session_id": session_id[:8] + "…",
            "user_id": user_id[:12] + "…",
            "entry_type": entry_type,
            "id": entry_id_raw,
            "top_k": top_k,
        },
    )
    if entry_type not in {"decision", "note"}:
        payload = {"success": False, "error": "entry_type должен быть decision или note"}
        _log_http_response("/debug/memory/long-term/delete", 400, payload)
        return jsonify(payload), 400
    try:
        entry_id = int(entry_id_raw)
    except (TypeError, ValueError):
        payload = {"success": False, "error": "id должен быть целым числом"}
        _log_http_response("/debug/memory/long-term/delete", 400, payload)
        return jsonify(payload), 400

    try:
        deleted = agent.memory.delete_long_term_entry(
            session_id=session_id,
            user_id=user_id,
            entry_type=entry_type,
            entry_id=entry_id,
        )
        snapshot = agent.memory.debug_snapshot(session_id=session_id, user_id=user_id, query=query, top_k=top_k)
        payload = {"success": True, "deleted": bool(deleted), "snapshot": snapshot}
        _log_http_response(
            "/debug/memory/long-term/delete",
            200,
            {
                "success": True,
                "deleted": bool(deleted),
                "entry_type": entry_type,
                "id": entry_id,
                "long_term_decisions": len(snapshot.get("long_term", {}).get("decisions_top_k") or []),
                "long_term_notes": len(snapshot.get("long_term", {}).get("notes_top_k") or []),
                "memory_writes": len(snapshot.get("memory_writes") or []),
            },
        )
        return jsonify(payload)
    except Exception as exc:
        logger.exception("[DEBUG] memory long-term delete failed: %s", exc)
        _log_http_response("/debug/memory/long-term/delete", 500, {"error": str(exc)})
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/debug/ctx-strategy", methods=["POST"])
def debug_ctx_strategy():
    """Переключает стратегию управления контекстом."""
    data = request.get_json(silent=True) or {}
    strategy = data.get("strategy", "sticky_facts")
    try:
        agent.ctx.set_strategy(strategy)
        save_ctx_state(_get_or_create_session(), agent.ctx.dump())
        return jsonify(
            {
                "success": True,
                "strategy": agent.ctx.active,
                "stats": agent.ctx.stats(agent.conversation_history),
            }
        )
    except ValueError as exc:
        return jsonify({"success": False, "error": str(exc)}), 400


@app.route("/ctx/checkpoint", methods=["POST"])
def ctx_checkpoint():
    """Создать checkpoint в текущей точке диалога."""
    data = request.get_json(silent=True) or {}
    if agent.ctx.active != "branching":
        return jsonify({"error": "Branching стратегия не активна"}), 400
    name = data.get("name", f"cp_{len(agent.ctx.strategy.checkpoints) + 1}")
    result = agent.ctx.strategy.create_checkpoint(name)
    save_ctx_state(_get_or_create_session(), agent.ctx.dump())
    return jsonify({"success": True, "checkpoint": result, "stats": agent.ctx.stats(agent.conversation_history)})


@app.route("/ctx/fork", methods=["POST"])
def ctx_fork():
    """Создать ветку от checkpoint."""
    data = request.get_json(silent=True) or {}
    if agent.ctx.active != "branching":
        return jsonify({"error": "Branching стратегия не активна"}), 400
    checkpoint_name = data.get("checkpoint")
    branch_name = data.get("branch_name")
    try:
        new_branch = agent.ctx.strategy.fork(checkpoint_name, branch_name)
        save_ctx_state(_get_or_create_session(), agent.ctx.dump())
        return jsonify(
            {
                "success": True,
                "new_branch": new_branch,
                "branches": agent.ctx.strategy.list_branches(),
                "stats": agent.ctx.stats(agent.conversation_history),
            }
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/ctx/switch-branch", methods=["POST"])
def ctx_switch_branch():
    """Переключиться на ветку."""
    data = request.get_json(silent=True) or {}
    if agent.ctx.active != "branching":
        return jsonify({"error": "Branching стратегия не активна"}), 400
    branch = data.get("branch")
    try:
        agent.ctx.strategy.switch_branch(branch)
        save_ctx_state(_get_or_create_session(), agent.ctx.dump())
        return jsonify(
            {
                "success": True,
                "active_branch": agent.ctx.strategy.active_branch,
                "stats": agent.ctx.stats(agent.conversation_history),
            }
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/session/restore", methods=["GET"])
def restore_session():
    """Восстанавливает состояние активного проекта пользователя."""
    user_id = _resolve_request_user_id()
    active_project = _ensure_active_project(user_id)
    requested_session = str(request.args.get("session_id") or "").strip()

    if requested_session:
        requested_project = get_project_by_session(user_id=user_id, session_id=requested_session)
        if requested_project:
            if not bool(requested_project.get("is_active")):
                storage_activate_project(user_id=user_id, project_id=str(requested_project.get("id") or ""))
            active_project = _ensure_active_project(user_id)

    if not active_project:
        payload = {"found": False, "needs_project": True}
        _log_http_response("/session/restore", 200, payload)
        response = jsonify(payload)
        return _set_session_cookie(response, None, user_id=user_id)

    session_id = str(active_project.get("session_id") or "").strip()
    _log_http_request(
        "/session/restore",
        {"session_id": session_id, "project_id": active_project.get("id"), "user_id": user_id[:12] + "…"},
    )

    data = _load_agent_runtime_session(session_id=session_id, user_id=user_id) or {}
    full_messages = data.get("messages", [])
    logger.info("[RESTORE] Сессия восстановлена: %s… сообщений=%s", session_id[:8], len(full_messages))

    ctx_state = _build_ctx_state()
    working_view = agent.memory.get_working_view(session_id=session_id)
    working_actions = agent.memory.get_working_actions(session_id=session_id)
    payload = {
        "found": True,
        "session_id": session_id,
        "project_id": active_project.get("id"),
        "project_name": active_project.get("name"),
        "current_model": agent.model,
        "available_models": agent.available_models(),
        "messages": full_messages,
        "token_stats_session": {
            "total_tokens_in": int(data.get("total_tokens_in", 0) or 0),
            "total_tokens_out": int(data.get("total_tokens_out", 0) or 0),
            "total_cost_usd": float(data.get("total_cost_usd", 0.0) or 0.0),
            "cost_history": data.get("cost_history", []),
        },
        "ctx_state": ctx_state,
        "ctx_stats": ctx_state["stats"],
        "ctx_strategy": ctx_state["strategy"],
        "working_view": working_view,
        "working_actions": working_actions,
    }
    _log_http_response(
        "/session/restore",
        200,
        {
            "found": payload["found"],
            "session_id": payload["session_id"],
            "project_name": payload["project_name"],
            "messages_count": len(payload["messages"]),
        },
    )
    response = jsonify(payload)
    return _set_session_cookie(response, session_id, user_id=user_id)


@app.route("/session/new", methods=["POST"])
def new_session():
    """Legacy endpoint: создаёт новый проект и активирует его."""
    data = request.get_json(silent=True) or {}
    user_id = _resolve_request_user_id(data)
    provided_name = str(data.get("name") or "").strip()
    name = provided_name or f"Проект {uuid4().hex[:6]}"
    if len(name) > 50:
        name = name[:50]
    project = create_project(user_id=user_id, name=name, activate=True)

    global _runtime_session_id
    _runtime_session_id = None
    payload = {"success": True, "session_id": project["session_id"], "project_id": project["id"], "name": project["name"]}
    response = jsonify(payload)
    _log_http_response("/session/new", 200, payload)
    return _set_session_cookie(response, project["session_id"], user_id=user_id)


@app.route("/reset", methods=["POST"])
def reset():
    """Очищает историю текущего активного проекта."""
    user_id = _resolve_request_user_id()
    active_project = _ensure_active_project(user_id)
    session_id = str(active_project.get("session_id") or "") if active_project else ""
    _log_http_request("/reset", {"session_id": session_id, "user_id": user_id[:12] + "…"})

    if session_id and session_exists(session_id):
        clear_session_messages(session_id)
        agent.clear_session_memory(session_id)

    global _runtime_session_id
    _runtime_session_id = None
    payload = {"success": True, "cleared": ["history"]}
    _log_http_response("/reset", 200, payload)
    response = jsonify(payload)
    return _set_session_cookie(response, session_id or None, user_id=user_id)


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    logger.info("[START] Сервер запущен → http://localhost:%s", port)
    app.run(debug=True, port=port)
