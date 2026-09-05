FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (curl, ffmpeg, OpenCV libs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy package manifests and source
COPY pyproject.toml README.md /app/
COPY animallens /app/animallens
COPY models /app/models

# Install CPU PyTorch wheel (fast, lightweight ~180MB) and vision dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -e ".[vision]"

# Expose conflict-free port 8088
EXPOSE 8088

# Health check against unauthenticated endpoint
HEALTHCHECK --interval=20s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8088/v1/health || exit 1

# Run AnimalLens FastAPI server
CMD ["python", "-m", "uvicorn", "animallens.server.app:app", "--host", "0.0.0.0", "--port", "8088"]
