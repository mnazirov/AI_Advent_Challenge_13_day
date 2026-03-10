from __future__ import annotations

import json
from typing import Any

from openai import BadRequestError, OpenAI

from llm.client import LLMChatResponse, LLMChoice, LLMMessage, LLMUsage


class OpenAILLMClient:
    """OpenAI-backed LLM client with model-compatibility fallbacks."""

    MODEL_COMPAT_PRESETS = {
        "gpt-5-mini": {"token_param": "max_completion_tokens", "drop_temperature": True},
    }

    def __init__(self, client: OpenAI | None = None):
        self.client = client or OpenAI()
        self._model_compat_overrides: dict[str, dict[str, Any]] = {}

    def chat_completion(self, **kwargs: Any) -> LLMChatResponse:
        request_kwargs = self._apply_model_compat(dict(kwargs))
        seen_signatures: set[tuple[tuple[str, str], ...]] = set()

        for _ in range(4):
            signature = tuple(sorted((k, repr(v)) for k, v in request_kwargs.items()))
            if signature in seen_signatures:
                break
            seen_signatures.add(signature)
            try:
                raw = self.client.chat.completions.create(**request_kwargs)
                return self._convert_response(raw)
            except BadRequestError as exc:
                fallback_kwargs = self._adapt_request_for_known_compat(request_kwargs, exc)
                if fallback_kwargs is None:
                    raise
                request_kwargs = fallback_kwargs

        raw = self.client.chat.completions.create(**request_kwargs)
        return self._convert_response(raw)

    def _convert_response(self, raw: Any) -> LLMChatResponse:
        content = ""
        finish_reason = None
        if getattr(raw, "choices", None):
            choice = raw.choices[0]
            finish_reason = getattr(choice, "finish_reason", None)
            message = getattr(choice, "message", None)
            parsed_payload = getattr(message, "parsed", None)
            if parsed_payload is not None:
                content = self._to_json_text(parsed_payload)
            raw_content = getattr(message, "content", "")
            if not content:
                content = self._coerce_content_text(raw_content)
            if not content:
                content = self._coerce_content_text(message)
            if not content:
                content = self._coerce_content_text(choice)
        if not content:
            content = self._coerce_content_text(raw)
        # Не подставлять имя роли как content при пустом ответе API (избегаем "assistant" вместо JSON)
        if content and content.strip().lower() in ("assistant", "user", "system"):
            content = ""

        usage_raw = getattr(raw, "usage", None)
        prompt_tokens = int(getattr(usage_raw, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage_raw, "completion_tokens", 0) or 0)
        total_tokens = int(getattr(usage_raw, "total_tokens", prompt_tokens + completion_tokens) or 0)

        return LLMChatResponse(
            id=getattr(raw, "id", None),
            model=getattr(raw, "model", None),
            choices=[LLMChoice(message=LLMMessage(content=content), finish_reason=finish_reason)],
            usage=LLMUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            ),
            raw=raw,
        )

    @staticmethod
    def _to_json_text(value: Any) -> str:
        try:
            return json.dumps(value, ensure_ascii=False)
        except Exception:
            return ""

    def _coerce_content_text(self, value: Any) -> str:
        return self._coerce_content_text_inner(value, visited=set())

    def _coerce_content_text_inner(self, value: Any, *, visited: set[int]) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, (bytes, bytearray)):
            try:
                return value.decode("utf-8")
            except Exception:
                return str(value)
        if isinstance(value, (int, float, bool)):
            return ""

        if isinstance(value, (list, tuple, set)):
            object_id = id(value)
            if object_id in visited:
                return ""
            visited.add(object_id)
            return "".join(self._coerce_content_text_inner(item, visited=visited) for item in value)

        if isinstance(value, dict):
            object_id = id(value)
            if object_id in visited:
                return ""
            visited.add(object_id)
            preferred_keys = (
                "text",
                "value",
                "output_text",
                "output",
                "input_text",
                "content",
                "message",
                "refusal",
                "reasoning",
                "summary",
                "arguments",
            )
            chunks: list[str] = []
            for key in preferred_keys:
                if key in value:
                    chunk = self._coerce_content_text_inner(value.get(key), visited=visited)
                    if chunk:
                        chunks.append(chunk)
            if chunks:
                return "".join(chunks)
            return "".join(self._coerce_content_text_inner(v, visited=visited) for v in value.values())

        object_id = id(value)
        if object_id in visited:
            return ""
        visited.add(object_id)

        for method_name in ("model_dump", "to_dict", "dict"):
            method = getattr(value, method_name, None)
            if callable(method):
                try:
                    dumped = method()
                except TypeError:
                    try:
                        dumped = method(exclude_none=True)
                    except Exception:
                        dumped = None
                except Exception:
                    dumped = None
                if dumped is not None and dumped is not value:
                    chunk = self._coerce_content_text_inner(dumped, visited=visited)
                    if chunk:
                        return chunk

        preferred_attrs = (
            "text",
            "value",
            "output_text",
            "output",
            "input_text",
            "content",
            "message",
            "refusal",
            "reasoning",
            "summary",
            "arguments",
        )
        chunks: list[str] = []
        for attr in preferred_attrs:
            attr_value = getattr(value, attr, None)
            if attr_value is None or attr_value is value:
                continue
            chunk = self._coerce_content_text_inner(attr_value, visited=visited)
            if chunk:
                chunks.append(chunk)
        if chunks:
            return "".join(chunks)

        return ""

    def _apply_model_compat(self, request_kwargs: dict[str, Any]) -> dict[str, Any]:
        adjusted = dict(request_kwargs)
        model_name = str(adjusted.get("model") or "")

        profile: dict[str, Any] = {}
        preset = self.MODEL_COMPAT_PRESETS.get(model_name)
        if isinstance(preset, dict):
            profile.update(preset)
        cached = self._model_compat_overrides.get(model_name)
        if isinstance(cached, dict):
            profile.update(cached)

        token_param = profile.get("token_param")
        if token_param == "max_completion_tokens" and "max_tokens" in adjusted and "max_completion_tokens" not in adjusted:
            adjusted["max_completion_tokens"] = adjusted.pop("max_tokens")
        elif token_param == "max_tokens" and "max_completion_tokens" in adjusted and "max_tokens" not in adjusted:
            adjusted["max_tokens"] = adjusted.pop("max_completion_tokens")

        if profile.get("drop_temperature"):
            adjusted.pop("temperature", None)
        if profile.get("drop_response_format"):
            adjusted.pop("response_format", None)

        return adjusted

    def _adapt_request_for_known_compat(self, request_kwargs: dict[str, Any], exc: BadRequestError) -> dict[str, Any] | None:
        error_text = str(exc).lower()
        fallback_kwargs = dict(request_kwargs)
        model_name = str(fallback_kwargs.get("model") or "")

        unsupported_param = "unsupported parameter" in error_text
        mentions_max_tokens = "max_tokens" in error_text
        mentions_max_completion = "max_completion_tokens" in error_text
        mentions_response_format = "response_format" in error_text
        if unsupported_param and (mentions_max_tokens or mentions_max_completion):
            if "max_tokens" in fallback_kwargs:
                self._remember_model_compat(model_name, token_param="max_completion_tokens")
                return self._apply_model_compat(fallback_kwargs)
            if "max_completion_tokens" in fallback_kwargs:
                self._remember_model_compat(model_name, token_param="max_tokens")
                return self._apply_model_compat(fallback_kwargs)
        if unsupported_param and mentions_response_format and "response_format" in fallback_kwargs:
            self._remember_model_compat(model_name, drop_response_format=True)
            return self._apply_model_compat(fallback_kwargs)

        unsupported_value = "unsupported value" in error_text
        temperature_issue = "temperature" in error_text
        if unsupported_value and temperature_issue and "temperature" in fallback_kwargs:
            self._remember_model_compat(model_name, drop_temperature=True)
            return self._apply_model_compat(fallback_kwargs)

        return None

    def _remember_model_compat(
        self,
        model_name: str,
        *,
        token_param: str | None = None,
        drop_temperature: bool | None = None,
        drop_response_format: bool | None = None,
    ) -> None:
        if not model_name:
            return
        profile = self._model_compat_overrides.setdefault(model_name, {})
        if token_param in {"max_tokens", "max_completion_tokens"}:
            profile["token_param"] = token_param
        if drop_temperature is True:
            profile["drop_temperature"] = True
        if drop_response_format is True:
            profile["drop_response_format"] = True
