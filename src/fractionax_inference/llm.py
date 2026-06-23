"""Single-shot completion with Claude -> MiniMax failover.

Anthropic (Claude) is primary. When Claude is unavailable — connection errors,
timeouts, rate limits, or 5xx/overloaded responses — the completion fails over to
MiniMax via its OpenAI-compatible API. Failover only triggers on *availability*
errors; client errors (bad request, auth) re-raise. Set ``MINIMAX_API_KEY`` to
enable the fallback (or run on MiniMax alone by leaving ``ANTHROPIC_API_KEY``
unset).
"""

from __future__ import annotations

import logging
from typing import Any, cast

import anthropic
from anthropic import Anthropic
from anthropic.types import MessageParam, TextBlock

from .config import Settings, get_settings

logger = logging.getLogger("fractionax_inference.llm")

_UNAVAILABLE_STATUS = {408, 409, 429, 500, 502, 503, 504, 529}
_NO_PROVIDER = "No LLM provider configured: set ANTHROPIC_API_KEY and/or MINIMAX_API_KEY."


def is_anthropic_unavailable(exc: Exception) -> bool:
    """True if ``exc`` means Claude is temporarily unavailable (failover-worthy)."""
    if isinstance(
        exc,
        (
            anthropic.APIConnectionError,
            anthropic.APITimeoutError,
            anthropic.RateLimitError,
            anthropic.InternalServerError,
        ),
    ):
        return True
    if isinstance(exc, anthropic.APIStatusError):
        return exc.status_code in _UNAVAILABLE_STATUS
    return False


def _complete_anthropic(settings: Settings, *, system: str, prompt: str) -> str:
    client = Anthropic(api_key=settings.anthropic_api_key)
    response = client.messages.create(
        model=settings.model,
        max_tokens=settings.max_tokens,
        system=system,
        messages=cast("list[MessageParam]", [{"role": "user", "content": prompt}]),
    )
    return "".join(b.text for b in response.content if isinstance(b, TextBlock))


def _complete_minimax(settings: Settings, *, system: str, prompt: str) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=settings.minimax_api_key, base_url=settings.minimax_base_url)
    response = client.chat.completions.create(
        model=settings.minimax_model,
        max_tokens=settings.max_tokens,
        messages=cast(
            "Any",
            [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        ),
    )
    return response.choices[0].message.content or ""


def complete(*, system: str, prompt: str) -> str:
    """Single-shot completion, failing over from Claude to MiniMax."""
    settings = get_settings()
    if settings.anthropic_api_key:
        try:
            return _complete_anthropic(settings, system=system, prompt=prompt)
        except Exception as exc:
            if not (settings.minimax_api_key and is_anthropic_unavailable(exc)):
                raise
            logger.warning("Claude unavailable (%s); failing over to MiniMax", type(exc).__name__)
    if settings.minimax_api_key:
        return _complete_minimax(settings, system=system, prompt=prompt)
    raise RuntimeError(_NO_PROVIDER)
