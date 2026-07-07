# syntax=docker/dockerfile:1

# Slim Python 3.12 base. uvicorn is launched with uvloop + httptools (installed
# via uvicorn[standard]) for the lowest-latency async event loop.
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install dependencies first for better layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code.
COPY app ./app
COPY scripts ./scripts

# Run as a non-root user.
RUN useradd --create-home --uid 10001 nawa \
    && chown -R nawa:nawa /app
USER nawa

EXPOSE 8000

# A single async worker: this is an I/O-bound streaming service, so one process
# on uvloop scales across many concurrent calls. Scale horizontally (more ECS
# tasks) rather than adding worker processes. --http httptools + --loop uvloop
# are the low-latency defaults from uvicorn[standard].
CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--loop", "uvloop", "--http", "httptools", \
     "--workers", "1", "--no-access-log", "--timeout-keep-alive", "75"]
