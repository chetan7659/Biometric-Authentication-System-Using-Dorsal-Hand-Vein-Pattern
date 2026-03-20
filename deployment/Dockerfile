# ─────────────────────────────────────────────────────────────────────────────
# Dorsal Hand Vein Biometric Auth — Production Dockerfile
# Target platforms: Railway.app · Docker Desktop · AWS EC2
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.9-slim

# ── Metadata ─────────────────────────────────────────────
LABEL maintainer="Chetan Sethy"
LABEL description="Dorsal Hand Vein Cancellable Biometric Auth — Streamlit App"
LABEL version="1.0"

# ── Working directory ─────────────────────────────────────
WORKDIR /app

# ── System dependencies ───────────────────────────────────
# libglib2.0-0 is required by some scipy/numpy internals on slim images.
# We use opencv-python-HEADLESS so libgl1-mesa-glx is NOT needed.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ───────────────────────────────────
# Copy requirements first for Docker layer caching.
# A rebuild only re-runs pip install if requirements.txt actually changed.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Application code ──────────────────────────────────────
# .dockerignore prevents large/unnecessary files from being copied.
COPY . .

# ── Streamlit config ──────────────────────────────────────
# .streamlit/config.toml is copied as part of COPY . .
# It configures headless mode, theme, and port for cloud deployment.

# ── Security: run as non-root user ───────────────────────
RUN useradd -m -u 1001 appuser && chown -R appuser:appuser /app
USER appuser

# ── Port ─────────────────────────────────────────────────
EXPOSE 8501

# ── Health check ─────────────────────────────────────────
# Railway, Docker, and Docker Compose all use this endpoint to verify
# the app is ready to serve traffic. Streamlit exposes it built-in.
HEALTHCHECK --interval=30s --timeout=15s --start-period=60s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# ── Start command ─────────────────────────────────────────
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
