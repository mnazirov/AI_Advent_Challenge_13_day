from __future__ import annotations

from typing import Any

from openai import BadRequestError, OpenAI

from llm.client import LLMChatResponse, LLMChoice, LLMMessage, LLMUsage


class OpenAILLMClient:
    """OpenAI-backed LLM client with model-compatibility fallbacks."""

    MODEL_COMPAT_PRESETS = {
        "gpt-5.3-instant": {"token_param": "max_completion_tokens", "drop_temperature": True},
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
            content = getattr(getattr(choice, "message", None), "content", "") or ""

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

        return adjusted

    def _adapt_request_for_known_compat(self, request_kwargs: dict[str, Any], exc: BadRequestError) -> dict[str, Any] | None:
        error_text = str(exc).lower()
        fallback_kwargs = dict(request_kwargs)
        model_name = str(fallback_kwargs.get("model") or "")

        unsupported_param = "unsupported parameter" in error_text
        mentions_max_tokens = "max_tokens" in error_text
        mentions_max_completion = "max_completion_tokens" in error_text
        if unsupported_param and (mentions_max_tokens or mentions_max_completion):
            if "max_tokens" in fallback_kwargs:
                self._remember_model_compat(model_name, token_param="max_completion_tokens")
                return self._apply_model_compat(fallback_kwargs)
            if "max_completion_tokens" in fallback_kwargs:
                self._remember_model_compat(model_name, token_param="max_tokens")
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
    ) -> None:
        if not model_name:
            return
        profile = self._model_compat_overrides.setdefault(model_name, {})
        if token_param in {"max_tokens", "max_completion_tokens"}:
            profile["token_param"] = token_param
        if drop_temperature is True:
            profile["drop_temperature"] = True
