# syntax=docker/dockerfile:1.7

ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        build-essential \
        ca-certificates \
        curl \
        git \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-optional.txt ./

ARG INSTALL_OPTIONAL=false
RUN python -m pip install --upgrade pip setuptools wheel \
    && pip install -r requirements.txt \
    && if [ "$INSTALL_OPTIONAL" = "true" ]; then pip install -r requirements-optional.txt; fi

ARG APP_UID=1000
ARG APP_GID=1000
RUN groupadd --gid ${APP_GID} alphagpt \
    && useradd --uid ${APP_UID} --gid alphagpt --create-home --home-dir /home/alphagpt --shell /bin/bash alphagpt \
    && chown alphagpt:alphagpt /app


FROM base AS dev

ARG INSTALL_DEV_TOOLS=true
RUN if [ "$INSTALL_DEV_TOOLS" = "true" ]; then \
        pip install ipython pytest ruff; \
    fi

COPY --chown=alphagpt:alphagpt . .

USER alphagpt
EXPOSE 8501

CMD ["bash"]


FROM base AS runtime

COPY --chown=alphagpt:alphagpt . .

USER alphagpt
EXPOSE 8501

# Override this at docker run time for the required workflow, for example:
#   python -m data_pipeline.run_pipeline
#   python -m model_core.engine
#   python -m strategy_manager.runner
#   streamlit run dashboard/app.py
CMD ["python", "-m", "model_core.engine"]
