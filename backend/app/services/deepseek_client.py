"""Minimal DeepSeek API client for connection and AI-analysis testing.

This client is intentionally isolated from the V3.6 model/champion/strategy
pipeline.  It only reads configuration, performs one-shot HTTP requests and
returns parsed results for the AI settings/test page.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import logging
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import requests

from backend.app.core.paths import resource_root


logger = logging.getLogger("deepseek")

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"
REQUEST_TIMEOUT_SECONDS = 60

ENV_KEY = "DEEPSEEK_API_KEY"
ENV_BASE_URL = "DEEPSEEK_BASE_URL"
ENV_MODEL = "DEEPSEEK_MODEL"

ERROR_API_KEY_INVALID = "API_KEY_INVALID"
ERROR_NETWORK = "NETWORK_ERROR"
ERROR_TIMEOUT = "TIMEOUT"
ERROR_MODEL_NOT_AVAILABLE = "MODEL_NOT_AVAILABLE"
ERROR_RATE_LIMITED = "RATE_LIMITED"
ERROR_INVALID_RESPONSE = "INVALID_RESPONSE"


class DeepSeekError(RuntimeError):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(detail)
        self.code = code
        self.detail = detail


@dataclass(frozen=True, slots=True)
class DeepSeekConfig:
    api_key: str
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL


def config_path() -> Path:
    return resource_root() / "config" / "deepseek.local.json"


def mask_key(api_key: str) -> str:
    if not api_key:
        return ""
    if len(api_key) <= 8:
        return "****"
    return f"{api_key[:3]}****{api_key[-4:]}"


def load_config() -> DeepSeekConfig:
    file_config: Mapping[str, Any] = {}
    path = config_path()
    if path.is_file():
        try:
            file_config = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("deepseek config file unreadable: %s", exc)
    api_key = os.environ.get(ENV_KEY) or str(file_config.get("api_key") or "")
    base_url = os.environ.get(ENV_BASE_URL) or str(
        file_config.get("base_url") or DEFAULT_BASE_URL
    )
    model = os.environ.get(ENV_MODEL) or str(
        file_config.get("model") or DEFAULT_MODEL
    )
    return DeepSeekConfig(
        api_key=api_key.strip(),
        base_url=base_url.strip().rstrip("/"),
        model=model.strip(),
    )


def save_config(
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
) -> DeepSeekConfig:
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, Any] = {}
    if path.is_file():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing = {}
    if api_key is not None:
        existing["api_key"] = api_key
    if base_url is not None:
        existing["base_url"] = base_url
    if model is not None:
        existing["model"] = model
    path.write_text(
        json.dumps(existing, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return load_config()


def _headers(config: DeepSeekConfig) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {config.api_key}",
        "Content-Type": "application/json",
    }


def _input_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


def _extract_json(content: str) -> Any:
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].strip().lower().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        text = text[start : end + 1]
    return json.loads(text)


def _log_request(
    config: DeepSeekConfig,
    content: str,
    started: float,
    status_code: int | None,
    ok: bool,
) -> None:
    logger.info(
        "deepseek call model=%s elapsed_ms=%d http_status=%s input_hash=%s ok=%s key=%s",
        config.model,
        int((time.monotonic() - started) * 1000),
        status_code,
        _input_hash(content),
        ok,
        mask_key(config.api_key),
    )


def _map_error(exc: requests.RequestException, status_code: int | None) -> DeepSeekError:
    if isinstance(exc, requests.Timeout):
        return DeepSeekError(ERROR_TIMEOUT, "request timed out after 60 seconds")
    if status_code in (401, 403):
        return DeepSeekError(ERROR_API_KEY_INVALID, "invalid or missing API key")
    if status_code == 404:
        return DeepSeekError(ERROR_MODEL_NOT_AVAILABLE, "model or endpoint not found")
    if status_code == 429:
        return DeepSeekError(ERROR_RATE_LIMITED, "rate limited by provider")
    return DeepSeekError(
        ERROR_NETWORK,
        f"network or upstream error: {type(exc).__name__}: {exc}",
    )


def _chat_once(
    config: DeepSeekConfig,
    *,
    system: str,
    user: str,
    max_tokens: int = 1024,
) -> tuple[str, int]:
    content, elapsed_ms, _usage = _chat_messages_once(
        config,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_tokens=max_tokens,
    )
    return content, elapsed_ms


def _chat_messages_once(
    config: DeepSeekConfig,
    *,
    messages: list[dict[str, str]],
    max_tokens: int = 1024,
) -> tuple[str, int, dict[str, Any]]:
    url = f"{config.base_url}/chat/completions"
    payload = {
        "model": config.model,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": max_tokens,
    }
    started = time.monotonic()
    log_input = "\n".join(
        str(message.get("content") or "")
        for message in messages
        if message.get("role") == "user"
    ) or str(messages[-1].get("content") or "") if messages else ""
    try:
        response = requests.post(
            url,
            headers=_headers(config),
            json=payload,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        _log_request(config, log_input, started, None, False)
        raise _map_error(exc, None) from exc
    status = response.status_code
    if status != 200:
        _log_request(config, log_input, started, status, False)
        raise _map_error(
            requests.RequestException(response.text[:500]),
            status,
        )
    try:
        data = response.json()
        content = str(data["choices"][0]["message"]["content"])
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        _log_request(config, log_input, started, status, False)
        raise DeepSeekError(
            ERROR_INVALID_RESPONSE,
            f"response JSON shape invalid: {type(exc).__name__}",
        ) from exc
    _log_request(config, log_input, started, status, True)
    usage = data.get("usage")
    if not isinstance(usage, dict):
        usage = {}
    return content, int((time.monotonic() - started) * 1000), dict(usage)


def chat_messages(
    config: DeepSeekConfig,
    *,
    system: str,
    messages: Sequence[Mapping[str, Any]],
    max_tokens: int = 1024,
) -> tuple[str, int]:
    """One-shot multi-turn chat with a system prompt (no retries)."""

    validated: list[dict[str, str]] = []
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if role not in ("user", "assistant") or not isinstance(content, str):
            raise DeepSeekError(
                ERROR_INVALID_RESPONSE,
                "chat messages must use role user|assistant and string content",
            )
        if not content.strip():
            raise DeepSeekError(
                ERROR_INVALID_RESPONSE, "chat message content must not be empty"
            )
        validated.append({"role": role, "content": content.strip()})
    if not validated:
        raise DeepSeekError(
            ERROR_INVALID_RESPONSE, "messages must not be empty"
        )
    content, elapsed_ms, _usage = _chat_messages_once(
        config,
        messages=[{"role": "system", "content": system}, *validated],
        max_tokens=max_tokens,
    )
    if not content.strip():
        raise DeepSeekError(
            ERROR_INVALID_RESPONSE, "model returned an empty reply"
        )
    return content, elapsed_ms


def test_connection(config: DeepSeekConfig) -> dict[str, Any]:
    content, elapsed_ms = _chat_once(
        config,
        system="You are a connectivity probe.",
        user="请只回复：DEEPSEEK_API_OK",
        max_tokens=32,
    )
    if "DEEPSEEK_API_OK" not in content:
        raise DeepSeekError(
            ERROR_INVALID_RESPONSE,
            f"unexpected probe response: {content[:120]!r}",
        )
    return {
        "ok": True,
        "model": config.model,
        "response": "DEEPSEEK_API_OK",
        "elapsed_ms": elapsed_ms,
    }


def chat_json(
    config: DeepSeekConfig,
    *,
    system: str,
    user: str,
) -> tuple[dict[str, Any], int]:
    parsed, elapsed_ms, _usage = chat_json_detailed(
        config,
        system=system,
        user=user,
    )
    return parsed, elapsed_ms


def chat_json_detailed(
    config: DeepSeekConfig,
    *,
    system: str,
    user: str,
    max_tokens: int = 2048,
) -> tuple[dict[str, Any], int, dict[str, Any]]:
    """Like chat_json but also returns provider token usage (may be empty)."""

    content, elapsed_ms, usage = _chat_messages_once(
        config,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_tokens=max_tokens,
    )
    try:
        parsed = _extract_json(content)
    except (json.JSONDecodeError, ValueError) as exc:
        raise DeepSeekError(
            ERROR_INVALID_RESPONSE,
            f"model returned non-JSON content: {exc}",
        ) from exc
    if not isinstance(parsed, dict):
        raise DeepSeekError(
            ERROR_INVALID_RESPONSE,
            "model returned a non-object JSON value",
        )
    return parsed, elapsed_ms, usage
