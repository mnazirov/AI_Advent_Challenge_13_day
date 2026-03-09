import asyncio
import logging
from dataclasses import dataclass, field

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MCPToolInfo:
    name: str
    description: str
    input_schema: dict = field(default_factory=dict)


@dataclass(frozen=True)
class MCPConnectionResult:
    success: bool
    tools: list[MCPToolInfo]
    error: str | None = None


@dataclass(frozen=True)
class MCPToolCallResult:
    success: bool
    tool: str
    arguments: dict = field(default_factory=dict)
    content: list[dict] = field(default_factory=list)
    structured_content: dict | None = None
    is_error: bool = False
    error: str | None = None


_SERVER_COMMAND = "python3"
_SERVER_ARGS = ["-m", "mcp_server_time", "--local-timezone", "Europe/Moscow"]


async def _fetch_tools() -> MCPConnectionResult:
    """
    Устанавливает stdio-соединение с MCP Time,
    инициализирует сессию, получает список инструментов.
    Не зависит от Flask. Не использует глобальное состояние.
    """
    server_params = StdioServerParameters(
        command=_SERVER_COMMAND,
        args=_SERVER_ARGS,
    )

    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                logger.info("[MCP_TIME] сессия инициализирована")

                response = await session.list_tools()
                logger.info("[MCP_TIME] получено инструментов: %s", len(response.tools))

                tools = [
                    MCPToolInfo(
                        name=tool.name,
                        description=tool.description or "",
                        input_schema=tool.inputSchema or {},
                    )
                    for tool in response.tools
                ]

                return MCPConnectionResult(success=True, tools=tools)
    except Exception as exc:
        logger.error("[MCP_TIME] ошибка соединения: %s", exc)
        return MCPConnectionResult(success=False, tools=[], error=str(exc))


def get_time_tools() -> MCPConnectionResult:
    """
    Синхронная точка входа. Вызывать из Flask-роутов.
    Каждый вызов создаёт новый event loop — соединение не кешируется,
    это намеренно для простоты MVP.
    """
    return asyncio.run(_fetch_tools())


async def _call_tool(tool: str, arguments: dict | None = None) -> MCPToolCallResult:
    """Вызывает конкретный инструмент MCP Time и возвращает нормализованный результат."""
    tool_name = str(tool or "").strip()
    args = arguments if isinstance(arguments, dict) else {}
    server_params = StdioServerParameters(
        command=_SERVER_COMMAND,
        args=_SERVER_ARGS,
    )

    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                logger.info("[MCP_TIME] call tool=%s args=%s", tool_name, list(args.keys()))

                response = await session.call_tool(name=tool_name, arguments=args)

                content: list[dict] = []
                for item in response.content or []:
                    if hasattr(item, "model_dump"):
                        content.append(item.model_dump(exclude_none=True, by_alias=True))
                    else:
                        content.append({"type": "unknown", "value": str(item)})

                return MCPToolCallResult(
                    success=True,
                    tool=tool_name,
                    arguments=args,
                    content=content,
                    structured_content=response.structuredContent,
                    is_error=bool(response.isError),
                )
    except Exception as exc:
        logger.error("[MCP_TIME] ошибка вызова инструмента %s: %s", tool_name, exc)
        return MCPToolCallResult(
            success=False,
            tool=tool_name,
            arguments=args,
            error=str(exc),
        )


def call_time_tool(tool: str, arguments: dict | None = None) -> MCPToolCallResult:
    """Синхронная точка входа для вызова MCP-инструмента из Flask."""
    return asyncio.run(_call_tool(tool, arguments or {}))
