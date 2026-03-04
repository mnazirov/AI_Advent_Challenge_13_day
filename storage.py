"""
storage.py — SQLite-хранилище сессий AI-ассистента.
Обеспечивает персистентность диалога, памяти и проектных данных между перезагрузками страницы.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from memory.models import LongTermProfile, ProfileConflict, ProfileField, ProfileSource, ProjectContext

logger = logging.getLogger("storage")

DB_PATH = Path("data/agent.db")
UPLOADS_DIR = Path("uploads")
SHORT_TERM_RUNTIME_ID = str(uuid.uuid4())


def init_db() -> None:
    """Инициализирует базу данных и создаёт таблицы если их нет."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

    with _get_conn() as conn:
        conn.executescript(
            """
            PRAGMA journal_mode=WAL;
            PRAGMA foreign_keys=ON;

            CREATE TABLE IF NOT EXISTS sessions (
                id          TEXT PRIMARY KEY,
                created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                -- TODO(cleanup): legacy upload fields kept for backward DB compatibility.
                filename    TEXT,
                csv_summary         TEXT,
                ctx_summary         TEXT    DEFAULT '',
                ctx_summarized_upto INTEGER DEFAULT 0,
                ctx_state           TEXT    DEFAULT '{}',
                schema_map          TEXT DEFAULT '{}',
                csv_path    TEXT,
                total_tokens_in  INTEGER DEFAULT 0,
                total_tokens_out INTEGER DEFAULT 0,
                total_cost_usd   REAL    DEFAULT 0.0,
                cost_history     TEXT    DEFAULT '[]'
            );

            CREATE TABLE IF NOT EXISTS projects (
                id          TEXT PRIMARY KEY,
                user_id     TEXT NOT NULL,
                name        TEXT NOT NULL,
                created_at  TEXT NOT NULL,
                session_id  TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                is_active   INTEGER DEFAULT 0
            );

            CREATE INDEX IF NOT EXISTS idx_projects_user_created
                ON projects(user_id, created_at);

            CREATE TABLE IF NOT EXISTS messages (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                role        TEXT NOT NULL CHECK(role IN ('user','assistant')),
                content     TEXT NOT NULL,
                created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_planning BOOLEAN DEFAULT 0,
                tokens_used INTEGER DEFAULT 0
            );

            CREATE INDEX IF NOT EXISTS idx_messages_session
                ON messages(session_id, created_at);

            CREATE TABLE IF NOT EXISTS memory_short_term_messages (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                runtime_id  TEXT NOT NULL,
                role        TEXT NOT NULL CHECK(role IN ('user','assistant','system')),
                content     TEXT NOT NULL,
                ts          TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_memory_short_term_session
                ON memory_short_term_messages(session_id, id);

            CREATE TABLE IF NOT EXISTS memory_working_tasks (
                session_id       TEXT PRIMARY KEY REFERENCES sessions(id) ON DELETE CASCADE,
                task_id          TEXT NOT NULL,
                goal             TEXT NOT NULL,
                state            TEXT NOT NULL,
                plan_json        TEXT NOT NULL DEFAULT '[]',
                current_step     TEXT NOT NULL DEFAULT '',
                done_steps_json  TEXT NOT NULL DEFAULT '[]',
                open_questions_json TEXT NOT NULL DEFAULT '[]',
                artifacts_json   TEXT NOT NULL DEFAULT '[]',
                vars_json        TEXT NOT NULL DEFAULT '{}',
                updated_at       TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS memory_longterm_profile (
                user_id          TEXT PRIMARY KEY,
                style            TEXT NOT NULL DEFAULT '',
                constraints_json TEXT NOT NULL DEFAULT '[]',
                context_json     TEXT NOT NULL DEFAULT '[]',
                tags_json        TEXT NOT NULL DEFAULT '[]',
                source           TEXT NOT NULL DEFAULT 'user',
                profile_json     TEXT NOT NULL DEFAULT '{}',
                profile_conflicts_json TEXT NOT NULL DEFAULT '[]',
                updated_at       TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS memory_longterm_decisions (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id    TEXT NOT NULL,
                type       TEXT NOT NULL DEFAULT 'decision',
                text       TEXT NOT NULL,
                tags_json  TEXT NOT NULL DEFAULT '[]',
                source     TEXT NOT NULL DEFAULT 'user',
                ttl_days   INTEGER,
                expires_at TEXT,
                created_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_memory_longterm_decisions_user
                ON memory_longterm_decisions(user_id, id DESC);

            CREATE TABLE IF NOT EXISTS memory_longterm_notes (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id    TEXT NOT NULL,
                type       TEXT NOT NULL DEFAULT 'note',
                text       TEXT NOT NULL,
                tags_json  TEXT NOT NULL DEFAULT '[]',
                source     TEXT NOT NULL DEFAULT 'user',
                ttl_days   INTEGER,
                expires_at TEXT,
                created_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_memory_longterm_notes_user
                ON memory_longterm_notes(user_id, id DESC);

            CREATE TABLE IF NOT EXISTS memory_longterm_pending (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id    TEXT NOT NULL,
                type       TEXT NOT NULL,
                text       TEXT NOT NULL,
                tags_json  TEXT NOT NULL DEFAULT '[]',
                source     TEXT NOT NULL DEFAULT 'assistant',
                ttl_days   INTEGER,
                status     TEXT NOT NULL DEFAULT 'pending',
                created_at TEXT NOT NULL,
                approved_at TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_memory_longterm_pending_user
                ON memory_longterm_pending(user_id, id DESC);
            """
        )
        _ensure_sessions_columns(conn)
        _ensure_projects_columns(conn)
        _ensure_memory_columns(conn)
        _cleanup_expired_longterm(conn)
        _cleanup_short_term_other_runtime(conn)

    cleaned = cleanup_old_sessions(30)
    logger.info("[STORAGE] БД инициализирована: %s | очищено_старых_сессий=%s", DB_PATH.resolve(), cleaned)


@contextmanager
def _get_conn():
    """Контекстный менеджер соединения с авто-commit и авто-close."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def create_session() -> str:
    """Создаёт новую сессию и возвращает её ID."""
    sid = str(uuid.uuid4())
    with _get_conn() as conn:
        conn.execute("INSERT INTO sessions (id) VALUES (?)", (sid,))
    logger.info("[STORAGE] Создана новая сессия: %s…", sid[:8])
    return sid


def ensure_session(session_id: str) -> None:
    """Гарантирует существование сессии с указанным ID."""
    sid = str(session_id or "").strip()
    if not sid:
        raise ValueError("session_id must be non-empty")
    with _get_conn() as conn:
        conn.execute("INSERT OR IGNORE INTO sessions (id) VALUES (?)", (sid,))


def session_exists(session_id: str) -> bool:
    """Проверяет существование сессии в БД."""
    with _get_conn() as conn:
        row = conn.execute("SELECT id FROM sessions WHERE id=?", (session_id,)).fetchone()
    return row is not None


def get_latest_session_id() -> str | None:
    """
    Возвращает ID последней активной сессии.
    Берётся сессия с максимальным updated_at.
    """
    with _get_conn() as conn:
        row = conn.execute(
            """
            SELECT id
            FROM sessions
            ORDER BY updated_at DESC
            LIMIT 1
            """
        ).fetchone()
    return row["id"] if row else None


def _project_row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "user_id": row["user_id"],
        "name": row["name"],
        "created_at": row["created_at"],
        "session_id": row["session_id"],
        "is_active": bool(row["is_active"]),
    }


def list_projects(user_id: str) -> list[dict[str, Any]]:
    """Возвращает проекты пользователя в порядке создания."""
    uid = str(user_id or "").strip()
    if not uid:
        return []
    with _get_conn() as conn:
        rows = conn.execute(
            """
            SELECT id, user_id, name, created_at, session_id, is_active
            FROM projects
            WHERE user_id=?
            ORDER BY created_at ASC, id ASC
            """,
            (uid,),
        ).fetchall()
    return [_project_row_to_dict(row) for row in rows]


def get_project(project_id: str, user_id: str) -> dict[str, Any] | None:
    """Возвращает проект пользователя по id."""
    pid = str(project_id or "").strip()
    uid = str(user_id or "").strip()
    if not pid or not uid:
        return None
    with _get_conn() as conn:
        row = conn.execute(
            """
            SELECT id, user_id, name, created_at, session_id, is_active
            FROM projects
            WHERE id=? AND user_id=?
            LIMIT 1
            """,
            (pid, uid),
        ).fetchone()
    if not row:
        return None
    return _project_row_to_dict(row)


def get_project_by_session(user_id: str, session_id: str) -> dict[str, Any] | None:
    """Возвращает проект пользователя по session_id."""
    uid = str(user_id or "").strip()
    sid = str(session_id or "").strip()
    if not uid or not sid:
        return None
    with _get_conn() as conn:
        row = conn.execute(
            """
            SELECT id, user_id, name, created_at, session_id, is_active
            FROM projects
            WHERE user_id=? AND session_id=?
            LIMIT 1
            """,
            (uid, sid),
        ).fetchone()
    if not row:
        return None
    return _project_row_to_dict(row)


def get_active_project(user_id: str) -> dict[str, Any] | None:
    """
    Возвращает активный проект пользователя.
    Если активного флага нет, активирует самый старый проект.
    """
    uid = str(user_id or "").strip()
    if not uid:
        return None
    with _get_conn() as conn:
        row = conn.execute(
            """
            SELECT id, user_id, name, created_at, session_id, is_active
            FROM projects
            WHERE user_id=? AND is_active=1
            ORDER BY created_at ASC, id ASC
            LIMIT 1
            """,
            (uid,),
        ).fetchone()
        if row:
            return _project_row_to_dict(row)

        fallback = conn.execute(
            """
            SELECT id, user_id, name, created_at, session_id, is_active
            FROM projects
            WHERE user_id=?
            ORDER BY created_at ASC, id ASC
            LIMIT 1
            """,
            (uid,),
        ).fetchone()
        if not fallback:
            return None

        conn.execute("UPDATE projects SET is_active=0 WHERE user_id=?", (uid,))
        conn.execute("UPDATE projects SET is_active=1 WHERE id=? AND user_id=?", (fallback["id"], uid))

    return {
        "id": fallback["id"],
        "user_id": fallback["user_id"],
        "name": fallback["name"],
        "created_at": fallback["created_at"],
        "session_id": fallback["session_id"],
        "is_active": True,
    }


def create_project(user_id: str, name: str, activate: bool = True) -> dict[str, Any]:
    """Создаёт проект и связанную chat-сессию."""
    uid = str(user_id or "").strip()
    project_name = str(name or "").strip()
    if not uid:
        raise ValueError("user_id must be non-empty")
    if not project_name:
        raise ValueError("name must be non-empty")
    if len(project_name) > 50:
        raise ValueError("name must be at most 50 characters")

    pid = str(uuid.uuid4())
    sid = str(uuid.uuid4())
    ts = datetime.utcnow().isoformat()

    with _get_conn() as conn:
        conn.execute("INSERT INTO sessions (id) VALUES (?)", (sid,))
        if activate:
            conn.execute("UPDATE projects SET is_active=0 WHERE user_id=?", (uid,))
        conn.execute(
            """
            INSERT INTO projects (id, user_id, name, created_at, session_id, is_active)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (pid, uid, project_name, ts, sid, 1 if activate else 0),
        )
    logger.info("[STORAGE] Создан проект: %s… user=%s… session=%s…", pid[:8], uid[:12], sid[:8])
    return {
        "id": pid,
        "user_id": uid,
        "name": project_name,
        "created_at": ts,
        "session_id": sid,
        "is_active": bool(activate),
    }


def activate_project(user_id: str, project_id: str) -> dict[str, Any] | None:
    """Делает указанный проект активным, остальные — неактивными."""
    uid = str(user_id or "").strip()
    pid = str(project_id or "").strip()
    if not uid or not pid:
        return None

    with _get_conn() as conn:
        row = conn.execute(
            """
            SELECT id, user_id, name, created_at, session_id, is_active
            FROM projects
            WHERE id=? AND user_id=?
            LIMIT 1
            """,
            (pid, uid),
        ).fetchone()
        if not row:
            return None

        conn.execute("UPDATE projects SET is_active=0 WHERE user_id=?", (uid,))
        conn.execute("UPDATE projects SET is_active=1 WHERE id=? AND user_id=?", (pid, uid))

    payload = _project_row_to_dict(row)
    payload["is_active"] = True
    logger.info("[STORAGE] Проект активирован: %s… user=%s…", pid[:8], uid[:12])
    return payload


def delete_project_with_session(user_id: str, project_id: str, delete_file: bool = True) -> dict[str, Any] | None:
    """Удаляет проект и связанную chat-сессию."""
    uid = str(user_id or "").strip()
    pid = str(project_id or "").strip()
    if not uid or not pid:
        return None

    project = get_project(project_id=pid, user_id=uid)
    if not project:
        return None

    session_id = str(project.get("session_id") or "").strip()
    if session_id:
        clear_session_csv(session_id, delete_file=delete_file)
        with _get_conn() as conn:
            conn.execute("DELETE FROM sessions WHERE id=?", (session_id,))
            conn.execute("DELETE FROM projects WHERE id=? AND user_id=?", (pid, uid))
    else:
        with _get_conn() as conn:
            conn.execute("DELETE FROM projects WHERE id=? AND user_id=?", (pid, uid))

    logger.info("[STORAGE] Проект удалён: %s… user=%s…", pid[:8], uid[:12])
    return project


def count_projects(user_id: str) -> int:
    """Возвращает количество проектов пользователя."""
    uid = str(user_id or "").strip()
    if not uid:
        return 0
    with _get_conn() as conn:
        row = conn.execute("SELECT COUNT(*) AS c FROM projects WHERE user_id=?", (uid,)).fetchone()
    return int(row["c"] if row else 0)


def save_csv_meta(
    session_id: str,
    filename: str,
    csv_summary: str,
    schema_map: dict,
    csv_path: str,
) -> None:
    """TODO(cleanup): сохраняет legacy upload-метаданные в сессию."""
    with _get_conn() as conn:
        conn.execute(
            """
            UPDATE sessions
            SET filename=?, csv_summary=?, schema_map=?, csv_path=?, updated_at=CURRENT_TIMESTAMP
            WHERE id=?
            """,
            (
                filename,
                csv_summary,
                json.dumps(schema_map, ensure_ascii=False),
                csv_path,
                session_id,
            ),
        )
    logger.info("[STORAGE] CSV-метаданные сохранены: сессия=%s… файл=%s", session_id[:8], filename)


def save_ctx_state(session_id: str, ctx_state: dict) -> None:
    """Сохраняет сериализованное состояние всех context-стратегий."""
    with _get_conn() as conn:
        conn.execute(
            "UPDATE sessions SET ctx_state=?, updated_at=CURRENT_TIMESTAMP WHERE id=?",
            (json.dumps(ctx_state, ensure_ascii=False), session_id),
        )
    logger.info("[STORAGE] ctx_state сохранён: сессия=%s…", session_id[:8])


def save_message(
    session_id: str,
    role: str,
    content: str,
    is_planning: bool = False,
    tokens_in: int = 0,
    tokens_out: int = 0,
    cost_usd: float = 0.0,
    tokens_used: int | None = None,
) -> None:
    """Сохраняет одно сообщение диалога."""
    message_tokens = int(tokens_used) if tokens_used is not None else int(tokens_in) + int(tokens_out)
    with _get_conn() as conn:
        conn.execute(
            """
            INSERT INTO messages (session_id, role, content, is_planning, tokens_used)
            VALUES (?, ?, ?, ?, ?)
            """,
            (session_id, role, content, int(bool(is_planning)), message_tokens),
        )
        if role == "assistant":
            current_cost_raw = conn.execute(
                "SELECT total_cost_usd, cost_history FROM sessions WHERE id=?",
                (session_id,),
            ).fetchone()
            if current_cost_raw:
                try:
                    cost_history = json.loads(current_cost_raw["cost_history"] or "[]")
                except json.JSONDecodeError:
                    cost_history = []
                if not isinstance(cost_history, list):
                    cost_history = []
                cost_history.append(float(cost_usd))
                conn.execute(
                    """
                    UPDATE sessions
                    SET total_tokens_in = total_tokens_in + ?,
                        total_tokens_out = total_tokens_out + ?,
                        total_cost_usd = total_cost_usd + ?,
                        cost_history = ?,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id=?
                    """,
                    (
                        int(tokens_in),
                        int(tokens_out),
                        float(cost_usd),
                        json.dumps(cost_history, ensure_ascii=False),
                        session_id,
                    ),
                )
            else:
                conn.execute("UPDATE sessions SET updated_at=CURRENT_TIMESTAMP WHERE id=?", (session_id,))
        else:
            conn.execute("UPDATE sessions SET updated_at=CURRENT_TIMESTAMP WHERE id=?", (session_id,))


def add_usage(
    session_id: str,
    tokens_in: int,
    tokens_out: int,
    cost_usd: float = 0.0,
) -> None:
    """
    Добавляет usage-токены/стоимость к агрегатам сессии без сохранения сообщения.
    Используется для LLM-вызовов вне чата (например, при загрузке CSV).
    """
    with _get_conn() as conn:
        current = conn.execute(
            "SELECT cost_history FROM sessions WHERE id=?",
            (session_id,),
        ).fetchone()
        cost_history = []
        if current:
            try:
                cost_history = json.loads(current["cost_history"] or "[]")
            except json.JSONDecodeError:
                cost_history = []
        if not isinstance(cost_history, list):
            cost_history = []
        if cost_usd > 0:
            cost_history.append(float(cost_usd))
        conn.execute(
            """
            UPDATE sessions
            SET total_tokens_in = total_tokens_in + ?,
                total_tokens_out = total_tokens_out + ?,
                total_cost_usd = total_cost_usd + ?,
                cost_history = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE id=?
            """,
            (
                int(tokens_in),
                int(tokens_out),
                float(cost_usd),
                json.dumps(cost_history, ensure_ascii=False),
                session_id,
            ),
        )


def load_session(session_id: str) -> dict | None:
    """Загружает полное состояние сессии для восстановления агента."""
    with _get_conn() as conn:
        row = conn.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
        if not row:
            return None

        msgs = conn.execute(
            """
            SELECT role, content, is_planning, tokens_used, created_at
            FROM messages
            WHERE session_id=?
            ORDER BY id DESC
            """,
            (session_id,),
        ).fetchall()

    ordered_msgs = list(reversed(msgs))
    return {
        "session_id": row["id"],
        "filename": row["filename"],
        "csv_summary": row["csv_summary"],
        "ctx_summary": row["ctx_summary"] or "",
        "ctx_summarized_upto": int(row["ctx_summarized_upto"] or 0),
        "ctx_state": json.loads(row["ctx_state"] or "{}"),
        "schema_map": json.loads(row["schema_map"] or "{}"),
        "csv_path": row["csv_path"],
        "total_tokens_in": int(row["total_tokens_in"] or 0),
        "total_tokens_out": int(row["total_tokens_out"] or 0),
        "total_cost_usd": float(row["total_cost_usd"] or 0.0),
        "cost_history": json.loads(row["cost_history"] or "[]"),
        "messages": [
            {
                "role": m["role"],
                "content": m["content"],
                "is_planning": bool(m["is_planning"]),
            }
            for m in ordered_msgs
        ],
    }


def memory_append_short_term_message(session_id: str, role: str, content: str, timestamp: str) -> None:
    """Добавляет сообщение в short-term память сессии."""
    ensure_session(session_id)
    with _get_conn() as conn:
        conn.execute(
            """
            INSERT INTO memory_short_term_messages (session_id, runtime_id, role, content, ts)
            VALUES (?, ?, ?, ?, ?)
            """,
            (session_id, SHORT_TERM_RUNTIME_ID, role, content, timestamp),
        )


def memory_load_short_term_messages(session_id: str) -> list[dict]:
    """Загружает short-term сообщения по сессии в порядке добавления."""
    with _get_conn() as conn:
        rows = conn.execute(
            """
            SELECT role, content, ts
            FROM memory_short_term_messages
            WHERE session_id=?
            ORDER BY id ASC
            """,
            (session_id,),
        ).fetchall()
    return [{"role": r["role"], "content": r["content"], "timestamp": r["ts"]} for r in rows]


def memory_load_short_term_messages_for_debug(session_id: str, limit_n: int) -> list[dict]:
    """Загружает последние limit_n short-term сообщений с id для debug-снимка (ORDER BY id ASC)."""
    with _get_conn() as conn:
        rows = conn.execute(
            """
            SELECT id, role, content, ts
            FROM (
                SELECT id, role, content, ts
                FROM memory_short_term_messages
                WHERE session_id=?
                ORDER BY id DESC
                LIMIT ?
            )
            ORDER BY id ASC
            """,
            (session_id, max(1, int(limit_n))),
        ).fetchall()
    return [
        {"id": int(r["id"]), "role": r["role"], "content": r["content"], "timestamp": r["ts"]}
        for r in rows
    ]


def memory_trim_short_term_messages(session_id: str, keep_last: int) -> int:
    """Оставляет только последние keep_last short-term сообщений и возвращает число удалённых."""
    keep = max(1, int(keep_last))
    with _get_conn() as conn:
        cur = conn.execute(
            """
            DELETE FROM memory_short_term_messages
            WHERE session_id=?
              AND id NOT IN (
                  SELECT id
                  FROM memory_short_term_messages
                  WHERE session_id=?
                  ORDER BY id DESC
                  LIMIT ?
              )
            """,
            (session_id, session_id, keep),
        )
    return int(cur.rowcount or 0)


def memory_clear_short_term_messages(session_id: str) -> None:
    """Удаляет всю short-term память сессии."""
    with _get_conn() as conn:
        conn.execute("DELETE FROM memory_short_term_messages WHERE session_id=?", (session_id,))


def _safe_json_loads(raw: str | None, default: object) -> object:
    try:
        return json.loads(raw or "")
    except Exception:
        return default


def _normalize_working_artifacts_for_storage(raw_items: list[dict] | list[str] | object) -> list[dict[str, str]]:
    items = raw_items if isinstance(raw_items, list) else []
    out: list[dict[str, str]] = []
    for item in items:
        if isinstance(item, dict):
            step = str(item.get("step") or "")
            ref = str(item.get("ref") or "")
            kind = str(item.get("type") or "legacy").strip().lower() or "legacy"
            if ref:
                out.append({"step": step, "type": kind, "ref": ref})
            continue

        if hasattr(item, "to_dict"):
            try:
                payload = item.to_dict()
            except Exception:
                payload = {}
            if isinstance(payload, dict):
                step = str(payload.get("step") or "")
                ref = str(payload.get("ref") or "")
                kind = str(payload.get("type") or "legacy").strip().lower() or "legacy"
                if ref:
                    out.append({"step": step, "type": kind, "ref": ref})
                continue

        ref = str(item or "").strip()
        if ref:
            out.append({"step": "", "type": "legacy", "ref": ref})
    return out


def memory_save_working_task(
    *,
    session_id: str,
    task_id: str,
    goal: str,
    state: str,
    plan: list[str],
    current_step: str | None,
    open_questions: list[str],
    artifacts: list[dict] | list[str],
    vars_data: dict,
    done_steps: list[str] | None = None,
    done: list[str] | None = None,
    updated_at: str | None = None,
    task: str | None = None,
) -> None:
    """Сохраняет рабочий контекст задачи для сессии."""
    ts = updated_at or datetime.utcnow().isoformat()
    effective_goal = str(task or goal or "").strip()
    done_payload = list(done if done is not None else (done_steps or []))
    artifacts_payload = _normalize_working_artifacts_for_storage(artifacts)
    with _get_conn() as conn:
        conn.execute(
            """
            INSERT INTO memory_working_tasks (
                session_id, task_id, goal, state,
                plan_json, current_step, done_steps_json, open_questions_json,
                artifacts_json, vars_json, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                task_id=excluded.task_id,
                goal=excluded.goal,
                state=excluded.state,
                plan_json=excluded.plan_json,
                current_step=excluded.current_step,
                done_steps_json=excluded.done_steps_json,
                open_questions_json=excluded.open_questions_json,
                artifacts_json=excluded.artifacts_json,
                vars_json=excluded.vars_json,
                updated_at=excluded.updated_at
            """,
            (
                session_id,
                task_id,
                effective_goal,
                state,
                json.dumps(plan, ensure_ascii=False),
                str(current_step or ""),
                json.dumps(done_payload, ensure_ascii=False),
                json.dumps(open_questions, ensure_ascii=False),
                json.dumps(artifacts_payload, ensure_ascii=False),
                json.dumps(vars_data, ensure_ascii=False),
                ts,
            ),
        )


def memory_load_working_task(session_id: str) -> dict | None:
    """Загружает рабочий контекст задачи по session_id."""
    with _get_conn() as conn:
        row = conn.execute(
            """
            SELECT *
            FROM memory_working_tasks
            WHERE session_id=?
            """,
            (session_id,),
        ).fetchone()
    if not row:
        return None
    raw_plan = _safe_json_loads(row["plan_json"], [])
    raw_done = _safe_json_loads(row["done_steps_json"], [])
    raw_open_questions = _safe_json_loads(row["open_questions_json"], [])
    raw_artifacts = _safe_json_loads(row["artifacts_json"], [])
    raw_vars = _safe_json_loads(row["vars_json"], {})

    plan = raw_plan if isinstance(raw_plan, list) else []
    done = raw_done if isinstance(raw_done, list) else []
    open_questions = raw_open_questions if isinstance(raw_open_questions, list) else []
    vars_data = raw_vars if isinstance(raw_vars, dict) else {}
    normalized_artifacts = _normalize_working_artifacts_for_storage(raw_artifacts)
    return {
        "session_id": row["session_id"],
        "task_id": row["task_id"],
        "task": row["goal"],
        "goal": row["goal"],
        "state": row["state"],
        "plan": plan,
        "current_step": row["current_step"] or None,
        "done": done,
        "done_steps": done,
        "open_questions": open_questions,
        "artifacts": normalized_artifacts,
        "vars": vars_data,
        "updated_at": row["updated_at"] or "",
    }


def memory_clear_working_task(session_id: str) -> None:
    """Удаляет рабочую память задачи по session_id."""
    with _get_conn() as conn:
        conn.execute("DELETE FROM memory_working_tasks WHERE session_id=?", (session_id,))


def clear_session_memory_layers(session_id: str) -> None:
    """Очищает только short-term и working память сессии."""
    with _get_conn() as conn:
        conn.execute("DELETE FROM memory_short_term_messages WHERE session_id=?", (session_id,))
        conn.execute("DELETE FROM memory_working_tasks WHERE session_id=?", (session_id,))


def memory_upsert_longterm_profile(
    *,
    user_id: str,
    profile: dict[str, Any] | LongTermProfile | None = None,
    conflicts: list[dict[str, Any]] | list[ProfileConflict] | None = None,
    style: str = "",
    constraints: list[str] | None = None,
    context: list[str] | dict[str, Any] | None = None,
    tags: list[str] | None = None,
    source: str = "user_explicit",
) -> None:
    """Создаёт или обновляет long-term профиль пользователя."""
    del tags
    profile_obj: LongTermProfile
    if profile is None:
        context_value = ProjectContext.from_any(context).to_dict()
        try:
            source_enum = ProfileSource(str(source or ProfileSource.USER_EXPLICIT.value))
        except ValueError:
            source_enum = ProfileSource.USER_EXPLICIT
        now_iso = datetime.utcnow().isoformat()
        profile_obj = LongTermProfile.default()
        profile_obj.response_style = ProfileField(
            value=str(style or ""),
            source=source_enum,
            verified=True,
            confidence=None,
            updated_at=now_iso,
        )
        profile_obj.hard_constraints = ProfileField(
            value=[str(x) for x in (constraints or [])],
            source=source_enum,
            verified=True,
            confidence=None,
            updated_at=now_iso,
        )
        profile_obj.project_context = ProfileField(
            value=context_value,
            source=source_enum,
            verified=True,
            confidence=None,
            updated_at=now_iso,
        )
        profile_obj.conflicts = []
    elif isinstance(profile, LongTermProfile):
        profile_obj = profile
    else:
        profile_obj = LongTermProfile.from_dict(profile)

    if conflicts is not None:
        normalized_conflicts: list[ProfileConflict] = []
        for item in conflicts:
            if isinstance(item, ProfileConflict):
                normalized_conflicts.append(item)
            elif isinstance(item, dict):
                normalized_conflicts.append(ProfileConflict.from_dict(item))
        profile_obj.conflicts = normalized_conflicts

    profile_payload = profile_obj.to_dict()
    profile_conflicts_payload = [c.to_dict() for c in profile_obj.conflicts]
    pc_value = ProjectContext.from_any(profile_obj.project_context.value).to_dict()
    legacy_style = str(profile_obj.response_style.value or "")
    legacy_constraints = [str(x) for x in (profile_obj.hard_constraints.value or [])]
    legacy_source = str(profile_obj.response_style.source.value or ProfileSource.USER_EXPLICIT.value)
    now_iso = datetime.utcnow().isoformat()

    with _get_conn() as conn:
        conn.execute(
            """
            INSERT INTO memory_longterm_profile (
                user_id, style, constraints_json, context_json, source, profile_json, profile_conflicts_json, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                style=excluded.style,
                constraints_json=excluded.constraints_json,
                context_json=excluded.context_json,
                source=excluded.source,
                profile_json=excluded.profile_json,
                profile_conflicts_json=excluded.profile_conflicts_json,
                updated_at=excluded.updated_at
            """,
            (
                user_id,
                legacy_style,
                json.dumps(legacy_constraints, ensure_ascii=False),
                json.dumps(pc_value, ensure_ascii=False),
                legacy_source,
                json.dumps(profile_payload, ensure_ascii=False),
                json.dumps(profile_conflicts_payload, ensure_ascii=False),
                now_iso,
            ),
        )


def memory_load_longterm_profile(user_id: str) -> dict | None:
    """Возвращает профиль long-term памяти пользователя."""
    with _get_conn() as conn:
        row = conn.execute(
            """
            SELECT
                user_id,
                style,
                constraints_json,
                context_json,
                tags_json,
                source,
                profile_json,
                profile_conflicts_json,
                updated_at
            FROM memory_longterm_profile
            WHERE user_id=?
            """,
            (user_id,),
        ).fetchone()
    if not row:
        return None

    raw_profile = row["profile_json"] or "{}"
    parsed_profile = _safe_json_loads(raw_profile, {})
    profile_obj: LongTermProfile
    if isinstance(parsed_profile, dict) and parsed_profile:
        profile_obj = LongTermProfile.from_dict(parsed_profile)
        raw_conflicts = _safe_json_loads(row["profile_conflicts_json"] or "[]", [])
        if isinstance(raw_conflicts, list) and raw_conflicts:
            profile_obj.conflicts = [ProfileConflict.from_dict(x) for x in raw_conflicts if isinstance(x, dict)]
    else:
        legacy_constraints = _safe_json_loads(row["constraints_json"], [])
        if not isinstance(legacy_constraints, list):
            legacy_constraints = []
        # TODO(migration): remove tags after DB migration confirmed in staging.
        legacy_context_raw = _safe_json_loads(row["context_json"], [])
        project_context = ProjectContext.from_any(legacy_context_raw).to_dict()
        now_iso = str(row["updated_at"] or datetime.utcnow().isoformat())
        source_raw = str(row["source"] or ProfileSource.USER_EXPLICIT.value)
        try:
            source_enum = ProfileSource(source_raw)
        except ValueError:
            source_enum = ProfileSource.USER_EXPLICIT
        profile_obj = LongTermProfile.default()
        profile_obj.response_style = ProfileField(
            value=str(row["style"] or ""),
            source=source_enum,
            verified=True,
            confidence=None,
            updated_at=now_iso,
        )
        profile_obj.hard_constraints = ProfileField(
            value=[str(x) for x in legacy_constraints],
            source=source_enum,
            verified=True,
            confidence=None,
            updated_at=now_iso,
        )
        profile_obj.project_context = ProfileField(
            value=project_context,
            source=source_enum,
            verified=True,
            confidence=None,
            updated_at=now_iso,
        )
        profile_obj.conflicts = []

    payload = profile_obj.to_dict()
    payload["user_id"] = row["user_id"]
    payload["updated_at"] = row["updated_at"] or ""
    payload["legacy_tags"] = _safe_json_loads(row["tags_json"], [])
    return payload


def memory_add_longterm_decision(
    *,
    user_id: str,
    text: str,
    tags: list[str],
    source: str = "user",
    ttl_days: int | None = None,
    entry_type: str = "decision",
) -> None:
    """Добавляет long-term решение."""
    expires_at = None
    if ttl_days is not None and int(ttl_days) > 0:
        expires_at = datetime.utcnow().timestamp() + int(ttl_days) * 86400
        expires_at = datetime.utcfromtimestamp(expires_at).isoformat()
    with _get_conn() as conn:
        conn.execute(
            """
            INSERT INTO memory_longterm_decisions (user_id, type, text, tags_json, source, ttl_days, expires_at, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                entry_type,
                text,
                json.dumps(tags, ensure_ascii=False),
                source,
                int(ttl_days) if ttl_days is not None else None,
                expires_at,
                datetime.utcnow().isoformat(),
            ),
        )


def memory_add_longterm_note(
    *,
    user_id: str,
    text: str,
    tags: list[str],
    source: str = "user",
    ttl_days: int | None = 90,
    entry_type: str = "note",
) -> None:
    """Добавляет long-term заметку."""
    expires_at = None
    if ttl_days is not None and int(ttl_days) > 0:
        expires_at = datetime.utcnow().timestamp() + int(ttl_days) * 86400
        expires_at = datetime.utcfromtimestamp(expires_at).isoformat()
    with _get_conn() as conn:
        conn.execute(
            """
            INSERT INTO memory_longterm_notes (user_id, type, text, tags_json, source, ttl_days, expires_at, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                entry_type,
                text,
                json.dumps(tags, ensure_ascii=False),
                source,
                int(ttl_days) if ttl_days is not None else None,
                expires_at,
                datetime.utcnow().isoformat(),
            ),
        )


def memory_list_longterm_decisions(user_id: str, limit: int = 50) -> list[dict]:
    """Возвращает список решений long-term памяти."""
    with _get_conn() as conn:
        rows = conn.execute(
            """
            SELECT id, user_id, type, text, tags_json, source, ttl_days, expires_at, created_at
            FROM memory_longterm_decisions
            WHERE user_id=?
              AND (expires_at IS NULL OR expires_at > ?)
            ORDER BY id DESC
            LIMIT ?
            """,
            (user_id, datetime.utcnow().isoformat(), int(limit)),
        ).fetchall()
    return [
        {
            "id": int(r["id"]),
            "user_id": r["user_id"],
            "type": r["type"] or "decision",
            "text": r["text"],
            "tags": json.loads(r["tags_json"] or "[]"),
            "source": r["source"],
            "ttl_days": r["ttl_days"],
            "expires_at": r["expires_at"],
            "created_at": r["created_at"],
        }
        for r in rows
    ]


def memory_list_longterm_notes(user_id: str, limit: int = 50) -> list[dict]:
    """Возвращает список заметок long-term памяти."""
    with _get_conn() as conn:
        rows = conn.execute(
            """
            SELECT id, user_id, type, text, tags_json, source, ttl_days, expires_at, created_at
            FROM memory_longterm_notes
            WHERE user_id=?
              AND (expires_at IS NULL OR expires_at > ?)
            ORDER BY id DESC
            LIMIT ?
            """,
            (user_id, datetime.utcnow().isoformat(), int(limit)),
        ).fetchall()
    return [
        {
            "id": int(r["id"]),
            "user_id": r["user_id"],
            "type": r["type"] or "note",
            "text": r["text"],
            "tags": json.loads(r["tags_json"] or "[]"),
            "source": r["source"],
            "ttl_days": r["ttl_days"],
            "expires_at": r["expires_at"],
            "created_at": r["created_at"],
        }
        for r in rows
    ]


def memory_delete_longterm_decision(user_id: str, decision_id: int) -> int:
    """Удаляет конкретное long-term решение пользователя. Возвращает число удалённых строк."""
    with _get_conn() as conn:
        cur = conn.execute(
            """
            DELETE FROM memory_longterm_decisions
            WHERE user_id=? AND id=?
            """,
            (user_id, int(decision_id)),
        )
    return int(cur.rowcount or 0)


def memory_delete_longterm_note(user_id: str, note_id: int) -> int:
    """Удаляет конкретную long-term заметку пользователя. Возвращает число удалённых строк."""
    with _get_conn() as conn:
        cur = conn.execute(
            """
            DELETE FROM memory_longterm_notes
            WHERE user_id=? AND id=?
            """,
            (user_id, int(note_id)),
        )
    return int(cur.rowcount or 0)


def memory_add_longterm_pending(
    *,
    user_id: str,
    entry_type: str,
    text: str,
    tags: list[str],
    source: str = "assistant",
    ttl_days: int | None = None,
) -> int:
    """Добавляет pending-запись long-term (требует подтверждения пользователя)."""
    with _get_conn() as conn:
        cur = conn.execute(
            """
            INSERT INTO memory_longterm_pending (
                user_id, type, text, tags_json, source, ttl_days, status, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, 'pending', ?)
            """,
            (
                user_id,
                entry_type,
                text,
                json.dumps(tags, ensure_ascii=False),
                source,
                int(ttl_days) if ttl_days is not None else None,
                datetime.utcnow().isoformat(),
            ),
        )
    return int(cur.lastrowid or 0)


def memory_list_longterm_pending(user_id: str, status: str = "pending", limit: int = 50) -> list[dict]:
    """Список pending-записей long-term."""
    with _get_conn() as conn:
        rows = conn.execute(
            """
            SELECT id, user_id, type, text, tags_json, source, ttl_days, status, created_at, approved_at
            FROM memory_longterm_pending
            WHERE user_id=? AND status=?
            ORDER BY id DESC
            LIMIT ?
            """,
            (user_id, status, int(limit)),
        ).fetchall()
    return [
        {
            "id": int(r["id"]),
            "user_id": r["user_id"],
            "type": r["type"],
            "text": r["text"],
            "tags": json.loads(r["tags_json"] or "[]"),
            "source": r["source"],
            "ttl_days": r["ttl_days"],
            "status": r["status"],
            "created_at": r["created_at"],
            "approved_at": r["approved_at"],
        }
        for r in rows
    ]


def memory_get_pending_by_id(user_id: str, pending_id: int) -> dict | None:
    """Возвращает pending-запись по id."""
    with _get_conn() as conn:
        row = conn.execute(
            """
            SELECT id, user_id, type, text, tags_json, source, ttl_days, status, created_at, approved_at
            FROM memory_longterm_pending
            WHERE user_id=? AND id=?
            """,
            (user_id, int(pending_id)),
        ).fetchone()
    if not row:
        return None
    return {
        "id": int(row["id"]),
        "user_id": row["user_id"],
        "type": row["type"],
        "text": row["text"],
        "tags": json.loads(row["tags_json"] or "[]"),
        "source": row["source"],
        "ttl_days": row["ttl_days"],
        "status": row["status"],
        "created_at": row["created_at"],
        "approved_at": row["approved_at"],
    }


def memory_mark_pending_approved(user_id: str, pending_id: int) -> None:
    """Отмечает pending-запись как подтверждённую."""
    with _get_conn() as conn:
        conn.execute(
            """
            UPDATE memory_longterm_pending
            SET status='approved', approved_at=?
            WHERE user_id=? AND id=?
            """,
            (datetime.utcnow().isoformat(), user_id, int(pending_id)),
        )


def clear_session_messages(session_id: str) -> None:
    """Очищает историю сообщений (legacy upload-метаданные сохраняются)."""
    with _get_conn() as conn:
        conn.execute("DELETE FROM messages WHERE session_id=?", (session_id,))
        conn.execute("DELETE FROM memory_short_term_messages WHERE session_id=?", (session_id,))
        conn.execute("DELETE FROM memory_working_tasks WHERE session_id=?", (session_id,))
        conn.execute(
            """
            UPDATE sessions
            SET total_tokens_in=0,
                total_tokens_out=0,
                total_cost_usd=0.0,
                cost_history='[]',
                ctx_summary='',
                ctx_summarized_upto=0,
                ctx_state='{}',
                updated_at=CURRENT_TIMESTAMP
            WHERE id=?
            """,
            (session_id,),
        )
    logger.info("[STORAGE] История очищена: сессия=%s…", session_id[:8])


def clear_session_csv(session_id: str, delete_file: bool = True) -> None:
    """
    TODO(cleanup): legacy helper from removed upload flow.
    Очищает привязанные upload-метаданные у сессии.
    При delete_file=True удаляет файл из uploads, если он существует.
    """
    csv_path = None
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT csv_path FROM sessions WHERE id=?",
            (session_id,),
        ).fetchone()
        if row:
            csv_path = row["csv_path"]

        conn.execute(
            """
            UPDATE sessions
            SET filename=NULL, csv_summary=NULL, schema_map='{}', csv_path=NULL, updated_at=CURRENT_TIMESTAMP
            WHERE id=?
            """,
            (session_id,),
        )

    if delete_file and csv_path:
        file_path = Path(csv_path)
        try:
            if file_path.exists():
                file_path.unlink()
                logger.info("[STORAGE] CSV-файл удалён: %s", file_path)
        except Exception as exc:
            logger.warning("[STORAGE] Не удалось удалить CSV-файл %s: %s", file_path, exc)

    logger.info("[STORAGE] CSV-данные очищены: сессия=%s…", session_id[:8])


def save_csv_file(file_content: bytes, filename: str) -> str:
    """TODO(cleanup): сохраняет legacy upload-файл на диск и возвращает путь к нему."""
    safe_name = Path(filename).stem[:40] + "_" + str(uuid.uuid4())[:8] + ".csv"
    filepath = UPLOADS_DIR / safe_name
    filepath.write_bytes(file_content)
    logger.info("[STORAGE] CSV сохранён: %s", filepath)
    return str(filepath)


def load_csv_file(csv_path: str) -> bytes | None:
    """TODO(cleanup): читает legacy upload-файл с диска."""
    path = Path(csv_path)
    if not path.exists():
        logger.warning("[STORAGE] CSV не найден на диске: %s", csv_path)
        return None
    return path.read_bytes()


def cleanup_old_sessions(days: int = 30) -> int:
    """Удаляет сессии старше N дней и возвращает их количество."""
    with _get_conn() as conn:
        _cleanup_expired_longterm(conn)
        result = conn.execute(
            """
            DELETE FROM sessions
            WHERE updated_at < datetime('now', ? || ' days')
            """,
            (f"-{int(days)}",),
        )
        conn.execute(
            """
            DELETE FROM memory_short_term_messages
            WHERE session_id NOT IN (SELECT id FROM sessions)
            """
        )
        conn.execute(
            """
            DELETE FROM memory_working_tasks
            WHERE session_id NOT IN (SELECT id FROM sessions)
            """
        )
        conn.execute(
            """
            DELETE FROM projects
            WHERE session_id NOT IN (SELECT id FROM sessions)
            """
        )
    deleted = result.rowcount or 0
    if deleted:
        logger.info("[STORAGE] Очищено старых сессий: %s", deleted)
    return int(deleted)


def _ensure_sessions_columns(conn: sqlite3.Connection) -> None:
    """Добавляет недостающие столбцы в sessions для старых БД."""
    rows = conn.execute("PRAGMA table_info(sessions)").fetchall()
    existing = {row["name"] for row in rows}
    migration = [
        ("total_tokens_in", "INTEGER DEFAULT 0"),
        ("total_tokens_out", "INTEGER DEFAULT 0"),
        ("total_cost_usd", "REAL DEFAULT 0.0"),
        ("cost_history", "TEXT DEFAULT '[]'"),
        ("ctx_summary", "TEXT DEFAULT ''"),
        ("ctx_summarized_upto", "INTEGER DEFAULT 0"),
        ("ctx_state", "TEXT DEFAULT '{}'"),
    ]
    for column_name, column_def in migration:
        if column_name not in existing:
            conn.execute(f"ALTER TABLE sessions ADD COLUMN {column_name} {column_def}")


def _ensure_projects_columns(conn: sqlite3.Connection) -> None:
    """Добавляет недостающие столбцы в projects для старых БД."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS projects (
            id          TEXT PRIMARY KEY,
            user_id     TEXT NOT NULL,
            name        TEXT NOT NULL,
            created_at  TEXT NOT NULL,
            session_id  TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
            is_active   INTEGER DEFAULT 0
        )
        """
    )
    rows = conn.execute("PRAGMA table_info(projects)").fetchall()
    existing = {row["name"] for row in rows}
    migration = [
        ("user_id", "TEXT NOT NULL DEFAULT ''"),
        ("name", "TEXT NOT NULL DEFAULT ''"),
        ("created_at", "TEXT NOT NULL DEFAULT ''"),
        ("session_id", "TEXT NOT NULL DEFAULT ''"),
        ("is_active", "INTEGER DEFAULT 0"),
    ]
    for column_name, column_def in migration:
        if column_name not in existing:
            conn.execute(f"ALTER TABLE projects ADD COLUMN {column_name} {column_def}")
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_projects_user_created
            ON projects(user_id, created_at)
        """
    )


def _ensure_memory_columns(conn: sqlite3.Connection) -> None:
    """Добавляет недостающие столбцы в memory-таблицы для старых БД."""
    checks = {
        "memory_short_term_messages": [
            ("runtime_id", "TEXT NOT NULL DEFAULT ''"),
        ],
        "memory_longterm_profile": [
            ("source", "TEXT NOT NULL DEFAULT 'user'"),
            ("profile_json", "TEXT NOT NULL DEFAULT '{}'"),
            ("profile_conflicts_json", "TEXT NOT NULL DEFAULT '[]'"),
        ],
        "memory_longterm_decisions": [
            ("type", "TEXT NOT NULL DEFAULT 'decision'"),
            ("ttl_days", "INTEGER"),
            ("expires_at", "TEXT"),
        ],
        "memory_longterm_notes": [
            ("type", "TEXT NOT NULL DEFAULT 'note'"),
            ("ttl_days", "INTEGER"),
            ("expires_at", "TEXT"),
        ],
    }

    for table_name, columns in checks.items():
        rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
        existing = {row["name"] for row in rows}
        for column_name, column_def in columns:
            if column_name not in existing:
                conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_def}")

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS memory_longterm_pending (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id    TEXT NOT NULL,
            type       TEXT NOT NULL,
            text       TEXT NOT NULL,
            tags_json  TEXT NOT NULL DEFAULT '[]',
            source     TEXT NOT NULL DEFAULT 'assistant',
            ttl_days   INTEGER,
            status     TEXT NOT NULL DEFAULT 'pending',
            created_at TEXT NOT NULL,
            approved_at TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_memory_longterm_pending_user
            ON memory_longterm_pending(user_id, id DESC)
        """
    )


def _cleanup_expired_longterm(conn: sqlite3.Connection) -> None:
    """Удаляет просроченные записи long-term по expires_at."""
    now = datetime.utcnow().isoformat()
    conn.execute(
        "DELETE FROM memory_longterm_decisions WHERE expires_at IS NOT NULL AND expires_at <= ?",
        (now,),
    )
    conn.execute(
        "DELETE FROM memory_longterm_notes WHERE expires_at IS NOT NULL AND expires_at <= ?",
        (now,),
    )


def _cleanup_short_term_other_runtime(conn: sqlite3.Connection) -> None:
    """
    Legacy no-op.

    Ранее short-term очищалась по runtime_id на каждом запуске приложения.
    Теперь short-term session-wide, поэтому cleanup по runtime_id отключён.
    """
    _ = conn
