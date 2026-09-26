"""DeepSeek AI settings/test endpoints (read-only with respect to models)."""

from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException, Request

from backend.app.services.deepseek_client import (
    DeepSeekError,
    chat_messages,
    load_config,
    mask_key,
    save_config,
    test_connection,
)
from backend.app.services.deepseek_market_payload import (
    analyze_market,
    build_market_context,
)


router = APIRouter(prefix="/api/ai/deepseek", tags=["ai-deepseek"])


def validate_chat_messages(raw_messages: object) -> list[dict[str, str]]:
    """Validate chat message payloads; return normalized user/assistant rows."""

    if not isinstance(raw_messages, list) or not raw_messages:
        raise ValueError("messages must be a non-empty list")
    messages: list[dict[str, str]] = []
    for item in raw_messages:
        if not isinstance(item, dict):
            raise ValueError("each message must be an object")
        role = item.get("role")
        content = item.get("content")
        if role not in ("user", "assistant"):
            raise ValueError("message role must be user or assistant")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("message content must be a non-empty string")
        messages.append({"role": role, "content": content.strip()})
    return messages


@router.get("/config")
def get_config() -> dict[str, object]:
    config = load_config()
    return {
        "has_key": bool(config.api_key),
        "base_url": config.base_url,
        "model": config.model,
    }


@router.put("/config")
def put_config(body: dict[str, object]) -> dict[str, object]:
    api_key = body.get("api_key")
    base_url = body.get("base_url")
    model = body.get("model")
    if api_key is not None and not isinstance(api_key, str):
        raise HTTPException(status_code=422, detail="api_key must be a string")
    if base_url is not None and not isinstance(base_url, str):
        raise HTTPException(status_code=422, detail="base_url must be a string")
    if model is not None and not isinstance(model, str):
        raise HTTPException(status_code=422, detail="model must be a string")
    config = save_config(
        api_key=api_key,
        base_url=base_url,
        model=model,
    )
    return {
        "has_key": bool(config.api_key),
        "base_url": config.base_url,
        "model": config.model,
        "masked_key": mask_key(config.api_key),
    }


@router.post("/test")
def test() -> dict[str, object]:
    config = load_config()
    if not config.api_key:
        raise HTTPException(
            status_code=422,
            detail="DEEPSEEK_API_KEY is not configured",
        )
    try:
        return test_connection(config)
    except DeepSeekError as exc:
        raise HTTPException(
            status_code=502,
            detail={"code": exc.code, "detail": exc.detail},
        ) from exc


@router.post("/analyze-market/{market}")
def analyze_market_endpoint(request: Request, market: str) -> dict[str, object]:
    config = load_config()
    if not config.api_key:
        raise HTTPException(
            status_code=422,
            detail="DEEPSEEK_API_KEY is not configured",
        )
    factory = getattr(request.app.state, "sessions", None)
    if factory is None:
        raise HTTPException(status_code=503, detail="session factory unavailable")
    try:
        with factory() as session:
            return analyze_market(session, market, config)
    except DeepSeekError as exc:
        raise HTTPException(
            status_code=502,
            detail={"code": exc.code, "detail": exc.detail},
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/chat")
def chat_endpoint(request: Request, body: dict[str, object]) -> dict[str, object]:
    config = load_config()
    if not config.api_key:
        raise HTTPException(
            status_code=422,
            detail="DEEPSEEK_API_KEY is not configured",
        )
    try:
        messages = validate_chat_messages(body.get("messages"))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    market = body.get("market")
    include_context = bool(body.get("include_market_context", True))
    system = (
        "你是行情分析助手。你可以基于用户提供或系统注入的只读行情数据回答问题；"
        "涉及未来判断时请说明依据与不确定性，不要声称保证收益。"
    )
    if include_context:
        if not market or not isinstance(market, str):
            raise HTTPException(
                status_code=422,
                detail="market is required when include_market_context is true",
            )
        factory = getattr(request.app.state, "sessions", None)
        if factory is None:
            raise HTTPException(status_code=503, detail="session factory unavailable")
        try:
            with factory() as session:
                context = build_market_context(session, market)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        system += (
            "\n\n当前只读行情上下文（供参考，不要当作事实承诺）：\n"
            + json.dumps(context, ensure_ascii=False, default=str)
        )
    try:
        reply, elapsed_ms = chat_messages(
            config,
            system=system,
            messages=messages,
        )
    except DeepSeekError as exc:
        raise HTTPException(
            status_code=502,
            detail={"code": exc.code, "detail": exc.detail},
        ) from exc
    return {
        "ok": True,
        "reply": reply,
        "elapsed_ms": elapsed_ms,
        "market": market if include_context else None,
    }
