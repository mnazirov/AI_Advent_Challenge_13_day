"""
Запуск: python3 -m unittest -v tests/test_mcp_duckduckgo.py
Или в общем прогоне: python3 -m unittest discover -s tests -v
Требует: server.py доступен (MCP_DDG_SERVER_PATH или ../AI_Advent_Challenge_17_day/server.py)
"""
import unittest

from mcp_duckduckgo import (
    MCPConnectionResult,
    MCPToolResult,
    call_define,
    call_related_topics,
    call_save_bookmark,
    call_search,
    call_search_bookmarks,
    get_ddg_capabilities,
)

EXPECTED_TOOLS = {"search", "define", "related_topics", "save_bookmark", "search_bookmarks"}
EXPECTED_RESOURCES = {"guide://search-tips", "bookmarks://all"}
EXPECTED_PROMPTS = {"research_prompt", "fact_check_prompt", "summarize_prompt"}


class TestMCPDDGCapabilities(unittest.TestCase):
    """Проверяет соединение и список capabilities."""

    @classmethod
    def setUpClass(cls):
        cls.caps: MCPConnectionResult = get_ddg_capabilities()

    def test_01_connection_success(self):
        self.assertTrue(self.caps.success, f"Соединение не установлено: {self.caps.error}")

    def test_02_all_tools_present(self):
        names = {tool.name for tool in self.caps.tools}
        for expected in EXPECTED_TOOLS:
            self.assertIn(expected, names, f"Tool '{expected}' не найден. Есть: {names}")

    def test_03_tools_have_descriptions(self):
        for tool in self.caps.tools:
            self.assertTrue(tool.description, f"Tool '{tool.name}' без описания")

    def test_04_resources_present(self):
        for resource in EXPECTED_RESOURCES:
            self.assertIn(resource, self.caps.resources, f"Resource '{resource}' не найден")

    def test_05_prompts_present(self):
        for prompt in EXPECTED_PROMPTS:
            self.assertIn(prompt, self.caps.prompts, f"Prompt '{prompt}' не найден")


class TestMCPDDGTools(unittest.TestCase):
    """Проверяет вызовы каждого tool."""

    def test_search_returns_result(self):
        result: MCPToolResult = call_search("Python programming")
        self.assertTrue(result.success, f"search упал: {result.error}")
        self.assertIsInstance(result.data, (dict, list))

    def test_search_empty_query_rejected(self):
        result = call_search("")
        self.assertFalse(result.success)
        self.assertIsNotNone(result.error)

    def test_define_returns_definition(self):
        result: MCPToolResult = call_define("API")
        self.assertTrue(result.success, f"define упал: {result.error}")
        self.assertIsInstance(result.data, dict)

    def test_related_topics_returns_result(self):
        result: MCPToolResult = call_related_topics("Python", limit=3)
        self.assertTrue(result.success, f"related_topics упал: {result.error}")

    def test_related_topics_invalid_limit(self):
        result = call_related_topics("test", limit=999)
        self.assertFalse(result.success)
        self.assertIsNotNone(result.error)

    def test_bookmark_save_and_search(self):
        save = call_save_bookmark(
            url="https://example.com",
            title="Example Site",
            tags=["test", "example"],
        )
        self.assertTrue(save.success, f"save_bookmark упал: {save.error}")

        search = call_search_bookmarks("example")
        self.assertTrue(search.success, f"search_bookmarks упал: {search.error}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
