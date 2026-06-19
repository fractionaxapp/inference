from __future__ import annotations

from typing import Any, cast

from anthropic import Anthropic
from anthropic.types import MessageParam, TextBlock

from .config import get_settings

SYSTEM_PROMPT = (
    "You are the FractionAX inference service. Answer the user's prompt directly "
    "and concisely. Do not invent facts."
)


def complete(prompt: str) -> str:
    """Single-shot Claude completion (no tools, no agentic loop)."""
    settings = get_settings()
    client = Anthropic(api_key=settings.anthropic_api_key)
    messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]

    response = client.messages.create(
        model=settings.model,
        max_tokens=settings.max_tokens,
        system=SYSTEM_PROMPT,
        messages=cast("list[MessageParam]", messages),
    )
    return "".join(b.text for b in response.content if isinstance(b, TextBlock))
