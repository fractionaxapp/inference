# @fractionax/inference

FractionAX model-inference service — [FastAPI](https://fastapi.tiangolo.com) + the
[Anthropic Claude](https://platform.claude.com) Python SDK, managed with
[uv](https://docs.astral.sh/uv/).

> **This repo is a submodule** of the [`ai`](https://github.com/fractionaxapp/ai)
> umbrella repo, itself a submodule of the
> [`fractionaxapp`](https://github.com/fractionaxapp/fractionaxapp) meta-monorepo.
> It is mounted at `ai/inference` and developed from the meta-repo.

A thin single-shot inference surface: one Claude completion per request (no tools,
no agentic loop — that's the `agents` service). Default model **Claude Opus 4.8**.
Exposes `GET /health` and `POST /v1/complete`.

## Develop (from the meta-repo root)

```bash
moon run inference:dev        # uvicorn (reload) — needs ANTHROPIC_API_KEY
moon run inference:test       # pytest
moon run inference:lint       # ruff check
moon run inference:typecheck  # mypy
```
