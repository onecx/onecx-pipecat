# syntax=docker/dockerfile:1
FROM ghcr.io/astral-sh/uv:0.9 AS uv

FROM python:3.12-slim-bookworm AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0 \
    UV_PROJECT_ENVIRONMENT=/app/.venv

COPY --from=uv /uv /uvx /bin/

WORKDIR /app

# Dependencies only, so this layer is cached until the lockfile changes.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-dev --no-install-project

# Bake NLTK's punkt_tab sentence-tokenizer data into the image so pipecat does
# not need to download it at runtime (the runtime user cannot write to it).
RUN /app/.venv/bin/python -c "import nltk; nltk.download('punkt_tab', download_dir='/app/nltk_data')"

FROM python:3.12-slim-bookworm AS runtime

ENV PATH=/app/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    ONECX_HOST=0.0.0.0 \
    NLTK_DATA=/app/nltk_data

RUN groupadd --system --gid 1001 app \
    && useradd --system --uid 1001 --gid app --no-create-home --shell /usr/sbin/nologin app

WORKDIR /app

COPY --from=builder --chown=1001:1001 /app/.venv /app/.venv
COPY --from=builder /app/nltk_data /app/nltk_data
COPY --chown=1001:1001 onecx_pipecat/ /app/onecx_pipecat/

USER 1001

EXPOSE 8080

# Host and port come from Settings (ONECX_HOST, SERVER_PORT / ONECX_PORT).
CMD ["python", "-m", "onecx_pipecat.main"]
