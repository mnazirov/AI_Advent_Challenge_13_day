# AI_Advent_Challenge_13_day

Локальный AI-ассистент для iOS-разработки с веб-интерфейсом, многоуровневой памятью и state machine задач.

Актуально по коду на `2026-03-10`.

## Что есть в проекте сейчас

- Flask API + web UI для диалога, управления проектами и debug-режима.
- Три слоя памяти: `short-term`, `working` (`PLANNING/EXECUTION/VALIDATION/DONE`) и `long-term`.
- Проектно-сессионная модель: у пользователя несколько проектов, у каждого проекта своя сессия, активный проект определяет контекст для `/chat`.
- Переключаемые стратегии контекста: `sliding_window`, `sticky_facts` (по умолчанию), `branching`, `history_compression`.
- Инварианты ответа (служебные маркеры, валидация переходов, нормализация).
- Debug API для просмотра и очистки отдельных слоёв памяти, а также редактирования профиля.

## Карта возможностей ассистента (для CLI)

Эта секция описывает не только endpoint-ы, но и фактическое поведение, которое CLI может использовать.

### 1) Диалог и управление состоянием задачи

- Автостарт рабочего контекста задачи из пользовательского запроса (task intent).
- Полноценная state machine с переходами: `PLANNING -> EXECUTION -> VALIDATION -> DONE` и откатом `VALIDATION -> EXECUTION`.
- Автопереход `EXECUTION -> VALIDATION`, когда закрыт последний шаг.
- На `VALIDATION` поддерживается explicit confirm и implicit confirm.
- В `DONE` возвращается итоговая сводка и она сохраняется в long-term notes.

### 2) Structured control через `client_intent`

CLI может управлять поведением ассистента структурированно, без reliance на NLP.

Поддерживаемые значения `client_intent.intent`:
- `task_intent`
- `plan_formation_intent`
- `decision_memory_write`
- `note_memory_write`
- `start_execution`
- `plan_approved`
- `plan_formation`
- `skip_mandatory_planning`
- `goal_clarification`
- `direct_code_request`
- `validation_request`
- `validation_checklist_request`
- `validation_confirm`
- `validation_reject`
- `validation_skip_request`
- `yes_confirmation`
- `no_confirmation`
- `stack_switch_request`
- `third_party_dependency_request`
- `step_completed`
- `working_update`
- `confirm_pending_memory`

Для CLI обычно полезны в первую очередь:
- `step_completed` (закрыть текущий шаг),
- `working_update` (передать patch в working memory),
- `validation_confirm` / `validation_reject`,
- `validation_request` / `validation_checklist_request`,
- `confirm_pending_memory`.

### 3) Shortcuts и спец-ответы ассистента

- Запросы воспоминаний о диалоге (`что я спрашивал`, `о чем мы говорили`) дают краткое резюме последних user-turns.
- В `EXECUTION` есть state-aware shortcut на вопрос `@StateObject` vs `@ObservedObject`.
- В `EXECUTION` запрос «покажи код/дай код» может возвращать готовый SwiftUI-snippet.
- В `VALIDATION` запрос чеклиста отдаёт финальный checklist.
- В `DONE` поддерживаются «что дальше» (next steps) и «сохрани в память» (форс-сохранение summary в long-term).

### 4) Ограничения и policy layer

- Hard constraints применяются из профиля только если поле `hard_constraints.verified == true`.
- В `PLANNING` и `EXECUTION` работает policy-блокировка попыток смены зафиксированного стека (например, SwiftUI -> React Native).
- В `EXECUTION` блокируются запросы на добавление сторонних зависимостей (с нативной альтернативой в ответе).
- Пропуск этапа `VALIDATION` запрещён.

### 5) Контекст и память

- Context strategies: `sliding_window`, `sticky_facts`, `branching`, `history_compression`.
- В `branching` доступны checkpoints/fork/switch branch через API.
- Layer-specific debug: snapshot всех слоёв памяти, очистка short/working/long-term по отдельности, удаление отдельных `decision`/`note`.
- Profile memory: canonical fields `stack_tools`, `response_style`, `hard_constraints`, `user_role_level`, `project_context`, а также `extra_fields` и `conflicts`; доступны операции add/update/delete/confirm/resolve conflict.

### 6) Диагностика для CLI-рендеринга

`POST /chat` возвращает служебные поля, полезные для TUI/CLI:
- `working_view` (state, current step, plan/done),
- `ctx_state` / `ctx_stats` / `ctx_strategy`,
- `memory_stats`,
- `token_stats`,
- `prompt_preview`,
- `invariant_report`,
- `finish_reason`.

Текущие `finish_reason`, которые имеет смысл обрабатывать в CLI:
- `stop`
- `length`
- `memory_recall`
- `invariant_fail`
- `state_blocked`
- `state_blocked_planning`
- `state_auto_validation`
- `state_auto_done`
- `state_done_error`
- `state_execution_context`
- `state_execution_code`
- `state_validation_checklist`
- `state_done_saved`
- `state_done_save_error`
- `state_done_next_steps`

Примечание:
- `working_actions` сейчас всегда возвращается как пустой список (`[]`) и зарезервирован под будущее UI-action API.

## Технологии

- Python `3.10+`
- Flask
- OpenAI Python SDK
- SQLite (`data/agent.db`)
- Опционально для `GET /mcp/time/tools`: `mcp` и `mcp_server_time` (не входят в `requirements.txt`).

`requirements.txt`:
- `openai`
- `flask`
- `pandas`
- `requests`

## Быстрый старт

1. Установите зависимости:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Создайте `.env`:

```bash
cp .env.example .env
```

3. Минимально заполните:

```env
OPENAI_API_KEY=your_openai_api_key
```

4. Запустите сервер:

```bash
python3 app.py
```

По умолчанию:
- URL: `http://localhost:5000`
- DB: `data/agent.db`
- Порт берётся из `PORT` (если не задан, используется `5000`).
- Сервер в текущем entrypoint запускается с `debug=True`.

Примечания по env:
- `OPENAI_MODEL` можно задать для стартовой модели (иначе `gpt-5.3-instant`).
- `DEEPSEEK_API_KEY` и `HUGGINGFACE_API_KEY` есть в `.env.example` как заготовка, но runtime этого проекта сейчас использует OpenAI-клиент.

## Поведение state machine

Состояния:
- `PLANNING`
- `EXECUTION`
- `VALIDATION`
- `DONE`

Фактические переходы:
- `PLANNING -> EXECUTION`: после подтверждения плана (включая семантическое подтверждение через классификатор).
- `EXECUTION -> VALIDATION`: автоматически после закрытия всех шагов плана.
- `VALIDATION -> EXECUTION`: при замечаниях/запросе доработки.
- `VALIDATION -> DONE`: при явном подтверждении, а также при implicit-confirm (если сообщение не распознано как reject).

Важный нюанс:
- на этапе `VALIDATION` нейтральное сообщение без признаков отклонения может трактоваться как согласие завершить задачу.

Ограничения этапов:
- в `PLANNING` блокируется прямой запрос «сразу дай код» при пустом плане.
- в `EXECUTION` запрещён пропуск `VALIDATION`.
- в `DONE` рабочая память заморожена для мутаций, но можно читать контекст и получать follow-up рекомендации.

## Модели

- Модель по умолчанию: `gpt-5.3-instant`.
- Список доступных моделей возвращает `GET /models`.
- Если запрошенная модель недоступна, агент делает fallback на `gpt-5-mini`, затем `gpt-4o-mini`.

## Основные endpoint-ы

Системные:
- `GET /` - web UI
- `GET /models` - текущая и доступные модели
- `GET /mcp/time/tools` - список инструментов MCP Time сервера
- `POST /model` - переключить модель

Проекты и сессии:
- `GET /projects`
- `POST /projects`
- `PATCH /projects/<project_id>/activate`
- `DELETE /projects/<project_id>`
- `GET /session/restore`
- `POST /session/new` (legacy-совместимость)
- `POST /reset`

Bootstrap-паттерн для CLI:
1. Вызвать `GET /session/restore?user_id=...`.
2. Если `found=true`, использовать `session_id` и `project_id`.
3. Если `found=false` и `needs_project=true`, создать проект через `POST /projects`.
4. Для stateless CLI лучше всегда явно передавать `user_id` в запросах, не полагаясь на cookie.

Ограничения по проектам:
- `POST /projects`: `name` обязателен, длина `<= 50` символов.
- `DELETE /projects/<project_id>`: нельзя удалить активный проект.
- `DELETE /projects/<project_id>`: нельзя удалить последний оставшийся проект пользователя.

Чат:
- `POST /chat`

`/chat` возвращает не только `reply`, но и служебные поля:
- `token_stats`
- `memory_stats`
- `ctx_state` / `ctx_stats` / `ctx_strategy`
- `working_view`
- `working_actions`
- `invariant_report`
- `finish_reason`

Контракт `POST /chat` (фактический):
- `message` (`string`) — обязателен и не должен быть пустым.
- `user_id` (`string`) — опционален (если не передан, берётся из cookie или создаётся).
- `session_id` (`string`) — опционален; если принадлежит проекту пользователя, может быть автоматически активирован.
- `model` (`string`) — опционален; при неподдерживаемом имени вернётся `400`.
- `client_intent` (`object`) — опциональная структурированная подсказка роутеру.

Формат `client_intent`:
- ключ интента: `intent` (также поддержаны `type`/`name`);
- `payload` — словарь параметров.

Пример:

```json
{
  "message": "Шаг выполнен",
  "client_intent": {
    "intent": "step_completed",
    "payload": {}
  }
}
```

Типичные ошибки `/chat`:
- `400` — нет активного проекта (`"Сначала создайте и выберите проект"`).
- `400` — пустое сообщение (`"Пустое сообщение"`).
- `400` — неподдерживаемая модель.

MCP Time:
- `GET /mcp/time/tools` возвращает `success`, `count`, `tools: [{name, description}]`.
- При проблеме соединения возвращает `500` и `{success:false,error,...}`.

Семантика `POST /reset`:
- очищает историю сообщений проекта в БД;
- очищает short-term и working слой текущей сессии;
- long-term память пользователя не удаляется.

Пример минимального сценария:

```bash
curl -X POST http://localhost:5000/projects \
  -H "Content-Type: application/json" \
  -d '{"user_id":"demo_user","name":"Demo Project"}'

curl -X POST http://localhost:5000/chat \
  -H "Content-Type: application/json" \
  -d '{"user_id":"demo_user","message":"Сформируй план задачи автоматически"}'
```

## Debug API

Память:
- `GET /debug/memory-layers`
- `POST /debug/memory/short-term/clear`
- `POST /debug/memory/working/clear`
- `POST /debug/memory/long-term/clear`
- `POST /debug/memory/long-term/delete`

Профиль:
- `GET /debug/memory/profile`
- `PATCH /debug/memory/profile/field`
- `POST /debug/memory/profile/field`
- `DELETE /debug/memory/profile/field`
- `POST /debug/memory/profile/confirm`
- `POST /debug/memory/profile/conflict/resolve`

Контекст-стратегии и ветвление:
- `POST /debug/ctx-strategy`
- `POST /ctx/checkpoint`
- `POST /ctx/fork`
- `POST /ctx/switch-branch`

Важно:
- `POST /ctx/checkpoint`, `POST /ctx/fork`, `POST /ctx/switch-branch` работают только при активной стратегии `branching`.
- Перед использованием этих endpoint-ов переключите стратегию через `POST /debug/ctx-strategy` с `{"strategy":"branching"}`.

## Структура проекта

- `app.py` - Flask-слой, HTTP API, проекты/сессии/cookies.
- `agent.py` - оркестратор диалога, инварианты, авто-переходы state machine.
- `memory/manager.py` - координация short/working/long-term.
- `memory/working.py` - строгие переходы состояния и операции по шагам.
- `memory/router.py` - intent-routing и извлечение изменений контекста.
- `memory/long_term.py` - профиль, решения, заметки, конфликты.
- `context_strategies.py` - менеджер стратегий контекста.
- `llm/openai_client.py` - OpenAI-адаптер с compat-fallback параметров.
- `storage.py` - SQLite-персистентность.
- `templates/index.html` - UI и debug drawer.
- `scripts/demo_memory_layers.py` - демонстрация слоёв памяти.

## Тесты

Основной прогон:

```bash
python3 -m unittest discover -s tests -v
```

Выборочно:

```bash
python3 -m unittest -v tests/test_working_memory.py
python3 -m unittest -v tests/test_state_machine_validation_done.py
python3 -m unittest -v tests/test_response_invariants.py
python3 -m unittest -v tests/test_mcp_time.py
```

`tests/test_mcp_time.py` проверяет интеграцию с MCP Time и требует установленных `mcp` и `mcp_server_time`.

Проверка на актуальном коде:
- `81` unit-тестов проходят успешно.

E2E сценарий state machine (требует запущенный сервер на `http://localhost:5000`):

```bash
pytest -q tests/test_state_machine_flow.py
```

Также можно запускать как обычный скрипт:

```bash
python3 tests/test_state_machine_flow.py
```

## Демо слоёв памяти

Mock-режим:

```bash
python3 scripts/demo_memory_layers.py --mock
```

Режим с реальным LLM:

```bash
python3 scripts/demo_memory_layers.py --real-llm
```

## Практические заметки

- Локальные данные пишутся в `data/` и `uploads/`.
- Куки `user_id` и `session_id` выставляются на 30 дней (`HttpOnly`).
- Для `/chat` нужен активный проект; если его нет, API вернёт ошибку с просьбой сначала создать/выбрать проект.
