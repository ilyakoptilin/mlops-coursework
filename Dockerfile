FROM python:3.12-slim

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
COPY models/model.joblib ./models/model.joblib

RUN uv sync --frozen --no-dev

ENV MODEL_PATH=models/model.joblib
ENV MODEL_VERSION=v1.0.0

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "mlops_coursework.api:app", "--host", "0.0.0.0", "--port", "8000"]
