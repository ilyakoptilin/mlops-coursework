FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /uvx /bin/

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY configs ./configs
COPY src ./src

RUN uv sync --frozen --no-dev && \
    python src/mlops_coursework/download_data.py && \
    python src/mlops_coursework/prepare_data.py && \
    python src/mlops_coursework/train.py && \
    rm -rf mlruns mlflow.db data/raw data/processed && \
    useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

CMD ["uvicorn", "mlops_coursework.api:app", "--host", "0.0.0.0", "--port", "8000"]
