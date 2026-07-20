# -----------------------------------------------------------------
# IntelliMoE -- Production Dockerfile
# Base image : python:3.11-slim
# Port       : 8501
# Note       : Uses CPU-only PyTorch (Groq + Gemini are cloud APIs,
#              no local GPU required). Image size ~1.5 GB vs ~4 GB.
# -----------------------------------------------------------------

FROM python:3.11-slim

# -- Environment hygiene ------------------------------------------
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# -- Working directory ---------------------------------------------
WORKDIR /app

# -- System dependencies ------------------------------------------
# Required by: ChromaDB (hnswlib), sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    gcc \
    g++ \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# -- Python dependencies ------------------------------------------
# Step 1: Install CPU-only PyTorch first (avoids pulling 1.5 GB of
#         CUDA/GPU binaries that are unused in this container).
RUN pip install --no-cache-dir \
    torch>=2.2.0 \
    --index-url https://download.pytorch.org/whl/cpu

# Step 2: Install remaining requirements
COPY requirements.txt .
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

# -- Application source -------------------------------------------
COPY . .

# -- Streamlit container configuration ----------------------------
RUN mkdir -p /root/.streamlit && \
    printf '[server]\nheadless = true\nenableCORS = false\nenableXsrfProtection = false\nfileWatcherType = "none"\n\n[browser]\ngatherUsageStats = false\n' \
    > /root/.streamlit/config.toml

# -- Port ---------------------------------------------------------
EXPOSE 8501

# -- Health check -------------------------------------------------
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# -- Entrypoint ---------------------------------------------------
# Pass API keys at runtime:
#   docker run --env-file .env -p 8501:8501 intellimoe
CMD ["streamlit", "run", "app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true"]