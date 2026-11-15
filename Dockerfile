# syntax=docker/dockerfile:1.7

FROM python:3.12-slim-bookworm

ARG TORCH_EXTRA=gpu
ARG AWS_CLI_VERSION=2.38.0

ENV PYTHONUNBUFFERED=1 \
    HYDRA_FULL_ERROR=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PATH="/app/.venv/bin:$PATH"

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        ca-certificates \
        curl \
        libegl1 \
        libgl1 \
        libglew2.2 \
        libgomp1 \
        libosmesa6 \
        unzip \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64-${AWS_CLI_VERSION}.zip" -o /tmp/awscliv2.zip \
    && unzip -q /tmp/awscliv2.zip -d /tmp \
    && /tmp/aws/install \
    && rm -rf /tmp/aws /tmp/awscliv2.zip

COPY --from=ghcr.io/astral-sh/uv:0.11.2 /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --extra "${TORCH_EXTRA}" --no-install-project

COPY weir ./weir
COPY configs ./configs
COPY cloud/entrypoint.sh ./cloud/entrypoint.sh

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --extra "${TORCH_EXTRA}"

RUN chmod +x /app/cloud/entrypoint.sh \
    && mkdir -p /work/run

ENTRYPOINT ["/app/cloud/entrypoint.sh"]
