# ─────────────────────────────────────────────────────────────────────────────
# Dockerfile  —  FastAPI Backend
# ─────────────────────────────────────────────────────────────────────────────
# WHAT THIS FILE IS:
#   A step-by-step recipe Docker follows to build an "image" — a snapshot
#   of a complete Linux system with Python, all packages, and our code.
#
# HOW DOCKER WORKS:
#   Image  = a frozen snapshot (like a ZIP of the whole system)
#   Container = a running instance of an image (like unzipping + executing)
#
#   You build the image ONCE.
#   You run containers FROM that image (can run many at once).
#
# EACH LINE EXPLAINED:
#   FROM    — which base image to start from (pre-built Linux + Python)
#   WORKDIR — sets the working directory inside the container
#   COPY    — copies files from your Mac into the container
#   RUN     — executes a command while building the image
#   ENV     — sets environment variables inside the container
#   EXPOSE  — documents which port the app listens on
#   CMD     — the command that runs when the container starts
# ─────────────────────────────────────────────────────────────────────────────

# ── Base Image ────────────────────────────────────────────────────────────────
# python:3.12-slim = official Python 3.12 on a minimal Debian Linux
# "slim" means no extra tools included — keeps the image small (~150MB vs 1GB)
FROM python:3.12-slim

# ── Build Arguments ───────────────────────────────────────────────────────────
# These can be overridden at build time: docker build --build-arg PORT=9000 .
ARG PORT=8000
ARG HOST=0.0.0.0

# ── Environment Variables ─────────────────────────────────────────────────────
# ENV variables are available both during build AND when the container runs.
# PYTHONUNBUFFERED=1  → Python prints output immediately (not buffered)
#                       Important for seeing logs in real time
# PYTHONDONTWRITEBYTECODE=1 → Don't create .pyc cache files (saves space)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    API_HOST=${HOST} \
    API_PORT=${PORT} \
    PYTHONPATH=/app

# ── Working Directory ─────────────────────────────────────────────────────────
# All subsequent commands run from /app inside the container.
# This is where our code will live.
WORKDIR /app

# ── System Dependencies ───────────────────────────────────────────────────────
# Some Python packages need C libraries installed at the OS level.
# apt-get is the Linux package manager (like Homebrew on Mac).
#
# build-essential → C compiler (needed for some Python packages)
# curl            → for health checks
# git             → sometimes needed by Hugging Face hub
#
# "&& rm -rf /var/lib/apt/lists/*" cleans up apt cache → smaller image
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# ── Install Python Dependencies ───────────────────────────────────────────────
# WHY COPY requirements.txt FIRST (before the rest of the code)?
#   Docker builds images in "layers" — each instruction is a layer.
#   Layers are CACHED. If requirements.txt hasn't changed, Docker reuses
#   the cached pip install layer instead of reinstalling everything.
#   This makes rebuilds after code changes MUCH faster (seconds vs minutes).
COPY requirements.txt .

# Install all Python packages
# --no-cache-dir → don't cache pip downloads (saves ~100MB in the image)
RUN pip install --upgrade pip && \
    pip install --no-cache-dir "setuptools==70.0.0" && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir "setuptools==70.0.0"

# ── Copy Application Code ─────────────────────────────────────────────────────
# Now copy the rest of our project files into the container.
# This layer changes every time code changes — that's why it comes AFTER pip install.
COPY app/ ./app/
COPY evaluation/ ./evaluation/
COPY .env .env

# ── Create Required Directories ───────────────────────────────────────────────
# These directories need to exist at runtime for the app to work.
# We create them here so they're ready when the container starts.
RUN mkdir -p data/uploads vector_store mlflow_tracking logs evaluation/results

# ── Expose Port ───────────────────────────────────────────────────────────────
# EXPOSE documents which port the container listens on.
# It doesn't actually open the port — docker run -p does that.
EXPOSE ${PORT}

# ── Health Check ──────────────────────────────────────────────────────────────
# Docker periodically runs this command to check if the container is healthy.
# If it fails 3 times in a row, Docker marks the container as "unhealthy".
# This is how orchestration tools (Kubernetes, ECS) know to restart a service.
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# ── Start Command ─────────────────────────────────────────────────────────────
# CMD is what runs when "docker run" starts the container.
# We use JSON array format (exec form) — more reliable than shell form.
# uvicorn is our ASGI server that runs the FastAPI app.
CMD ["uvicorn", "app.backend.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--log-level", "info"]
