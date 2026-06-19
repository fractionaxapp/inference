from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .config import get_settings
from .inference import complete

app = FastAPI(title="FractionAX Inference", version="0.0.0")


class CompleteRequest(BaseModel):
    prompt: str


class CompleteResponse(BaseModel):
    completion: str


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/complete", response_model=CompleteResponse)
def complete_endpoint(request: CompleteRequest) -> CompleteResponse:
    settings = get_settings()
    if not settings.anthropic_api_key:
        raise HTTPException(
            status_code=503,
            detail="Inference not configured: missing ANTHROPIC_API_KEY",
        )
    return CompleteResponse(completion=complete(request.prompt))
