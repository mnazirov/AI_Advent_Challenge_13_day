"""
MCP-клиент для DuckDuckGo сервера (AI_Advent_Challenge_17_day/server.py).
Транспорт: stdio. Пользователь запускает сервер сам: python3 server.py
"""
import asyncio
import json as _json
import logging
import os
from dataclasses import dataclass, field

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Гарантировать вывод независимо от конфигурации Flask
logger = logging.getLogger("mcp_duckduckgo")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)s | mcp_ddg | %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    logger.addHandler(_handler)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False


# Путь к server.py настраивается через env MCP_DDG_SERVER_PATH.
# По умолчанию ищем в директории-сестре AI_Advent_Challenge_17_day.
_DEFAULT_SERVER_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "AI_Advent_Challenge_17_day",
    "server.py",
)
SERVER_PATH = os.environ.get("MCP_DDG_SERVER_PATH", _DEFAULT_SERVER_PATH)
_SERVER_COMMAND = "python3"
_SERVER_ARGS = [SERVER_PATH]

# Диагностика при старте — выводится один раз при импорте модуля
_path_ok = os.path.exists(SERVER_PATH)
print(
    f"[MCP_DDG] server path: {SERVER_PATH}",
    f"\n[MCP_DDG] server exists: {_path_ok}",
    flush=True,
)
if not _path_ok:
    print(
        f"[MCP_DDG] WARNING: server.py не найден по пути {SERVER_PATH}\n"
        f"[MCP_DDG] Задайте MCP_DDG_SERVER_PATH=/путь/до/server.py в .env",
        flush=True,
    )


@dataclass(frozen=True)
class MCPToolInfo:
    name: str
    description: str
    input_schema: dict = field(default_factory=dict)


@dataclass(frozen=True)
class MCPToolResult:
    success: bool
    tool: str
    data: dict | list
    error: str | None = None


@dataclass(frozen=True)
class MCPConnectionResult:
    success: bool
    tools: list[MCPToolInfo]
    resources: list[str] = field(default_factory=list)
    prompts: list[str] = field(default_factory=list)
    error: str | None = None


def _server_params() -> StdioServerParameters:
    return StdioServerParameters(command=_SERVER_COMMAND, args=_SERVER_ARGS)


async def _fetch_capabilities() -> MCPConnectionResult:
    """Подключается к MCP-серверу, возвращает tools / resources / prompts."""
    logger.debug("[_FETCH] command=%s args=%s", _SERVER_COMMAND, _SERVER_ARGS)
    try:
        async with stdio_client(_server_params()) as (read, write):
            logger.debug("[_FETCH] stdio connected")
            async with ClientSession(read, write) as session:
                await session.initialize()
                logger.debug("[_FETCH] session initialized")

                tools_resp = await session.list_tools()
                resources_resp = await session.list_resources()
                prompts_resp = await session.list_prompts()

                tools = [
                    MCPToolInfo(
                        name=t.name,
                        description=t.description or "",
                        input_schema=t.inputSchema or {},
                    )
                    for t in tools_resp.tools
                ]
                resources = [str(resource.uri) for resource in resources_resp.resources]
                prompts = [prompt.name for prompt in prompts_resp.prompts]

                logger.info(
                    "[MCP_DDG] tools=%s resources=%s prompts=%s",
                    len(tools),
                    len(resources),
                    len(prompts),
                )
                return MCPConnectionResult(
                    success=True,
                    tools=tools,
                    resources=resources,
                    prompts=prompts,
                )
    except Exception as exc:
        logger.error("[MCP_DDG] ошибка соединения: %s", exc)
        return MCPConnectionResult(success=False, tools=[], error=str(exc))


async def _call_tool(name: str, arguments: dict) -> MCPToolResult:
    """
    Вызывает tool на MCP-сервере через stdio.
    Каждый вызов — новое соединение (MVP, без connection pool).
    """
    logger.debug("[_CALL] tool=%s args=%s", name, arguments)
    try:
        async with stdio_client(_server_params()) as (read, write):
            logger.debug("[_CALL] stdio connected for tool=%s", name)
            async with ClientSession(read, write) as session:
                await session.initialize()
                logger.debug("[_CALL] session initialized for tool=%s", name)
                response = await session.call_tool(name, arguments)
                logger.debug("[_CALL] response received for tool=%s", name)
                logger.info("[MCP_DDG] tool=%s ok", name)

                data: dict | list | None = None
                structured = getattr(response, "structuredContent", None)
                if isinstance(structured, (dict, list)):
                    if (
                        isinstance(structured, dict)
                        and set(structured.keys()) == {"result"}
                        and isinstance(structured.get("result"), (dict, list))
                    ):
                        data = structured["result"]
                    else:
                        data = structured

                if data is None:
                    content = response.content or []
                    text_items = [
                        item.text.strip()
                        for item in content
                        if hasattr(item, "text") and str(item.text or "").strip()
                    ]
                    if text_items:
                        raw = text_items[0]
                        try:
                            parsed = _json.loads(raw)
                            data = parsed if isinstance(parsed, (dict, list)) else {"text": raw}
                        except _json.JSONDecodeError:
                            data = {"text": raw}
                    else:
                        data = {}

                return MCPToolResult(success=True, tool=name, data=data)
    except Exception as exc:
        logger.error("[MCP_DDG] tool=%s error: %s", name, exc)
        return MCPToolResult(success=False, tool=name, data={}, error=str(exc))


def get_ddg_capabilities() -> MCPConnectionResult:
    logger.info("[CAPS] server_path=%s", SERVER_PATH)
    logger.info("[CAPS] server_exists=%s", os.path.exists(SERVER_PATH))
    result = asyncio.run(_fetch_capabilities())
    logger.info("[CAPS] success=%s tools=%s error=%s", result.success, len(result.tools), result.error)
    return result


def call_search(query: str) -> MCPToolResult:
    logger.info("[SEARCH] query=%r", query)
    if not query or not query.strip():
        logger.warning("[SEARCH] rejected: empty query")
        return MCPToolResult(
            success=False,
            tool="search",
            data={},
            error="query не может быть пустым",
        )
    result = asyncio.run(_call_tool("search", {"query": query.strip()}))
    logger.info(
        "[SEARCH] success=%s error=%s data_keys=%s",
        result.success,
        result.error,
        list(result.data.keys()) if isinstance(result.data, dict) else type(result.data).__name__,
    )
    return result


def call_define(term: str) -> MCPToolResult:
    logger.info("[DEFINE] term=%r", term)
    if not term or not term.strip():
        logger.warning("[DEFINE] rejected: empty term")
        return MCPToolResult(
            success=False,
            tool="define",
            data={},
            error="term не может быть пустым",
        )
    result = asyncio.run(_call_tool("define", {"term": term.strip()}))
    logger.info("[DEFINE] success=%s error=%s", result.success, result.error)
    return result


def call_related_topics(query: str, limit: int = 5) -> MCPToolResult:
    logger.info("[RELATED] query=%r limit=%r", query, limit)
    if not 1 <= limit <= 20:
        logger.warning("[RELATED] rejected: invalid limit=%r", limit)
        return MCPToolResult(
            success=False,
            tool="related_topics",
            data={},
            error="limit должен быть в диапазоне 1..20",
        )
    result = asyncio.run(_call_tool("related_topics", {"query": query.strip(), "limit": limit}))
    logger.info("[RELATED] success=%s error=%s", result.success, result.error)
    return result


def call_save_bookmark(url: str, title: str, tags: list[str] | None = None) -> MCPToolResult:
    logger.info("[BOOKMARK_SAVE] url=%r title=%r tags=%r", url, title, tags)
    result = asyncio.run(
        _call_tool(
            "save_bookmark",
            {"url": url.strip(), "title": title.strip(), "tags": tags or []},
        )
    )
    logger.info("[BOOKMARK_SAVE] success=%s error=%s", result.success, result.error)
    return result


def call_search_bookmarks(query: str) -> MCPToolResult:
    logger.info("[BOOKMARK_SEARCH] query=%r", query)
    result = asyncio.run(_call_tool("search_bookmarks", {"query": query.strip()}))
    logger.info("[BOOKMARK_SEARCH] success=%s error=%s", result.success, result.error)
    return result
