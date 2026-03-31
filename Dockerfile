# BioPAT Benchmark — CPU-only Docker image
# Multi-stage build for smaller image size

# Stage 1: Build dependencies
FROM python:3.12-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Install CPU-only PyTorch first (saves ~3GB vs CUDA version)
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu

# Copy project files for dependency resolution
COPY pyproject.toml .
COPY src/ src/

# Install project with evaluation + llm extras
RUN pip install --no-cache-dir -e ".[evaluation,llm]"

# Stage 2: Runtime image
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy project source
COPY . .

# Install in editable mode (source already copied)
RUN pip install --no-cache-dir --no-deps -e "."

# Create data directories
RUN mkdir -p data/raw data/processed data/cache data/benchmark data/checkpoints data/results logs

# Default environment
ENV BIOPAT_CPU_ONLY=true
ENV PYTHONUNBUFFERED=1

# Default command: show help
CMD ["python", "-m", "biopat.pipeline", "--help"]
