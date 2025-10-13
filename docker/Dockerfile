# docker/Dockerfile  (runtime)
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# install only runtime deps (no dev tools)
COPY requirements/base.txt ./requirements/base.txt
RUN python -m pip install -U pip && pip install -r requirements/base.txt

# copy code and assets for runtime
COPY src/ ./src/
COPY experiments/ ./experiments/

ENV PYTHONPATH=/app/src

# When CLI exists, enable:
# ENTRYPOINT ["python","-m","aiis.cli"]
# CMD ["--help"]