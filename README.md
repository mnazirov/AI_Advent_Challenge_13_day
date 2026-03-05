# AI_Advent_Challenge_13_day

Локальный AI-ассистент для iOS-продуктовой разработки (Swift/SwiftUI) с:
- web UI на Flask
- многоуровневой памятью (short-term, working, long-term)
- state machine задачи (`PLANNING -> EXECUTION -> VALIDATION -> DONE`)
- debug-интерфейсом для профиля, протокола и памяти

## Что умеет проект

- Вести диалог с LLM и сохранять историю по сессиям/проектам.
- Поддерживать рабочий контекст задачи с планом шагов.
- Автоматически переводить задачу в `VALIDATION`, когда все шаги выполнены.
- Требовать явного подтверждения для перехода `VALIDATION -> DONE`.
- Автоматически отправлять итоговые сообщения при входе в `VALIDATION` и `DONE`.
- Хранить долгосрочные решения и заметки пользователя.
- Показывать отладочные слои памяти и состояние protocol v2.

## Архитектура (кратко)

- `app.py`: Flask API, маршруты, восстановление сессий, связь с фронтом.
- `agent.py`: оркестрация хода диалога, вызов LLM, авто-реакции на переходы состояния.
- `memory/working.py`: строгая модель состояния задачи и переходов.
- `memory/router.py`: извлечение намерений из user message и запись в память.
- `memory/manager.py`: гейты состояния, actions для UI, debug-операции.
- `memory/long_term.py`: профиль, решения, заметки и pending-подтверждения.
- `memory/protocol.py`: protocol v2 (phase, invariants, next_step).
- `storage.py`: SQLite слой (`data/agent.db`).
- `templates/index.html`: UI и debug-панель.

## State Machine задачи

Состояния:
- `PLANNING`
- `EXECUTION`
- `VALIDATION`
- `DONE`

Ключевая логика:
- `PLANNING -> EXECUTION`: после явного подтверждения плана (`план утверждён`).
- `EXECUTION -> VALIDATION`: автоматически, когда закрыт последний шаг плана.
- `VALIDATION -> DONE`: только по явному подтверждению.
- `VALIDATION -> EXECUTION`: по сообщению о доработке.

Паттерны подтверждения `VALIDATION -> DONE`:
- `подтверждаю завершение`
- `всё готово`
- `задача выполнена`
- `переходим к итогам`
- `confirm`
- `готово`

Паттерны возврата `VALIDATION -> EXECUTION`:
- `вернуться к выполнению`
- `нужно доработать`
- `есть замечания`
- `back to execution`

## Требования

- Python 3.10+
- OpenAI API key

Пакеты:
- `openai`
- `flask`
- `pandas`
- `requests`

## Быстрый старт

1. Создайте виртуальное окружение и установите зависимости:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Создайте `.env` (можно на основе `.env.example`) и задайте ключ:

```bash
cp .env.example .env
```

Минимально нужен:

```env
OPENAI_API_KEY=your_openai_api_key
```

3. Запустите сервер:

```bash
python3 app.py
```

По умолчанию:
- URL: `http://localhost:5000`
- БД: `data/agent.db`

## Основной API

### Системные

- `GET /` - web UI
- `GET /models` - текущая модель и список доступных
- `POST /model` - переключение модели

### Проекты и сессии

- `GET /projects` - список проектов пользователя
- `POST /projects` - создать проект и активировать
- `PATCH /projects/<project_id>/activate` - активировать проект
- `DELETE /projects/<project_id>` - удалить проект (не активный и не последний)
- `GET /session/restore` - восстановить активную сессию
- `POST /session/new` - legacy-создание проекта/сессии
- `POST /reset` - очистить историю активного проекта

### Чат

- `POST /chat` - основной endpoint общения с ассистентом

Пример минимального сценария:

```bash
curl -X POST http://localhost:5000/projects \
  -H "Content-Type: application/json" \
  -d '{"user_id":"demo_user","name":"Demo Project"}'

curl -X POST http://localhost:5000/chat \
  -H "Content-Type: application/json" \
  -d '{"user_id":"demo_user","message":"Составь план задачи"}'
```

## Debug API

Ключевые endpoint-ы:
- `GET /debug/memory-layers`
- `GET /debug/memory/profile`
- `PATCH|POST|DELETE /debug/memory/profile/field`
- `POST /debug/memory/profile/confirm`
- `POST /debug/memory/profile/conflict/resolve`
- `POST /debug/memory/working/clear`
- `POST /debug/memory/long-term/delete`
- `GET /debug/protocol/status`
- `POST /debug/protocol/profile/confirm-update`
- `POST /debug/protocol/validate-type2`
- `POST /debug/ctx-strategy`
- `POST /ctx/checkpoint`
- `POST /ctx/fork`
- `POST /ctx/switch-branch`

## Стратегии контекста

Доступные стратегии (`context_strategies.py`):
- `sliding_window`
- `sticky_facts` (по умолчанию)
- `branching`
- `history_compression`

Переключение во время работы:
- `POST /debug/ctx-strategy`

## Тесты

Запуск всех тестов:

```bash
python3 -m unittest discover -s tests -v
```

Запуск выборочно:

```bash
python3 -m unittest -v tests/test_working_memory.py
python3 -m unittest -v tests/test_state_machine_validation_done.py
```

## Демо памяти

Скрипт:

```bash
python3 scripts/demo_memory_layers.py --mock
```

Режим с реальным LLM:

```bash
python3 scripts/demo_memory_layers.py --real-llm
```

## Полезные заметки

- Локальные артефакты пишутся в `data/` и `uploads/`.
- Если модель недоступна, агент пытается fallback на `gpt-5-mini`, затем `gpt-4o-mini`.
- В `DONE` рабочая память замораживается от записи, итог генерируется до заморозки.
