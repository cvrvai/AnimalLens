FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (curl, ffmpeg)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy package files
COPY pyproject.toml README.md /app/
COPY animallens /app/animallens

# Install AnimalLens package
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e .

# Expose API port
EXPOSE 8088

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8088/v1/health || exit 1

# Start server
CMD ["animallens", "serve", "--host", "0.0.0.0", "--port", "8088"]
