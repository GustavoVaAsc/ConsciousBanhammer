FROM nvidia/cuda:13.2.1-cudnn-runtime-ubuntu24.04@sha256:1c0c68dbf3258d32b446a02cb4be05c8478b65d320d7856ad33c4bbdf898ca86

ARG HOST_UID=1000
ARG HOST_GID=1000
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends ca-certificates git python3.12 \
    && rm -rf /var/lib/apt/lists/* \
    && userdel --remove ubuntu \
    && groupadd --gid ${HOST_GID} work \
    && useradd --create-home --uid ${HOST_UID} --gid ${HOST_GID} work \
    && mkdir /opt/venv /workspace \
    && chown work:work /opt/venv /workspace

COPY --from=ghcr.io/astral-sh/uv:0.12.0@sha256:606e70c71c852d03f611b1e56a195d08648507018a7057fab82c4974c4eae105 /uv /uvx /bin/

USER work

ENV HOME=/home/work \
    HF_HOME=/home/work/.cache/huggingface \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON=/usr/bin/python3.12 \
    UV_PYTHON_DOWNLOADS=never \
    PATH=/opt/venv/bin:$PATH

WORKDIR /workspace

COPY --chown=work:work pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/home/work/.cache/uv,uid=${HOST_UID},gid=${HOST_GID} \
    uv sync --locked --no-install-project --extra gpu

ENV UV_LOCKED=1

RUN mkdir -p ${HF_HOME}

CMD ["bash"]
