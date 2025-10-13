FROM mcr.microsoft.com/devcontainers/python:3.11

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git graphviz && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# copy only requirement lists first for better Docker layer caching
COPY requirements/base.txt requirements/dev.txt ./requirements/
RUN pip install -U pip && pip install -r requirements/base.txt -r requirements/developer.txt

ENV PYTHONPATH=/workspace/src