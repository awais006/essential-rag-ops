# syntax=docker/dockerfile:1

# ---- Builder: resolve deps + install the project into a venv with uv ----
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0
WORKDIR /app

# Layer 1 — dependencies only (cached unless uv.lock / pyproject.toml change)
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-dev

# Layer 2 — copy source, then install the project itself
COPY . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# ---- Runtime: slim image with just the venv + source, non-root ----
FROM python:3.12-slim-bookworm
RUN useradd --create-home --uid 1000 app
WORKDIR /app
COPY --from=builder --chown=app:app /app /app
ENV PATH="/app/.venv/bin:$PATH" \
    API_HOST=0.0.0.0 \
    API_PORT=8000
USER app
EXPOSE 8000
CMD ["rag-serve"]