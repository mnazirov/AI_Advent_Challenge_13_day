"""
Запуск: python3 -m unittest -v tests/test_mcp_time.py
Или в общем прогоне: python3 -m unittest discover -s tests -v
Проверяет соединение с MCP Time и список инструментов.
Не требует запущенного Flask-сервера.
"""
import unittest

from mcp_time import MCPConnectionResult, get_time_tools


class TestMCPTimeConnection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Одно соединение на все тесты класса."""
        cls.result: MCPConnectionResult = get_time_tools()

    def test_01_connection_success(self):
        self.assertTrue(self.result.success, f"Соединение не установлено: {self.result.error}")

    def test_02_tools_not_empty(self):
        self.assertGreater(len(self.result.tools), 0, "Список инструментов пуст")

    def test_03_expected_tool_present(self):
        names = {tool.name for tool in self.result.tools}
        self.assertIn("get_current_time", names, f"get_current_time не найден. Есть: {names}")

    def test_04_tools_have_descriptions(self):
        for tool in self.result.tools:
            self.assertTrue(tool.description, f"Инструмент '{tool.name}' без описания")

    def test_05_print_all_tools(self):
        """Информационный тест — выводит список для ручной проверки."""
        print("\n-- Список инструментов MCP Time --")
        for tool in self.result.tools:
            print(f"  * {tool.name}: {tool.description}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
