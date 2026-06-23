from __future__ import annotations

from . import llm

SYSTEM_PROMPT = (
    "You are the FractionAX inference service. Answer the user's prompt directly "
    "and concisely. Do not invent facts."
)


def complete(prompt: str) -> str:
    """Single-shot completion, failing over from Claude to MiniMax when unavailable."""
    return llm.complete(system=SYSTEM_PROMPT, prompt=prompt)
