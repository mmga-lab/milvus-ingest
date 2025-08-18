# Multi-stage build for optimized container size
FROM python:3.13.7-slim as builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install PDM
RUN pip install --upgrade pip && pip install pdm

# Copy project files
WORKDIR /app
COPY pyproject.toml pdm.lock README.md ./
COPY src/ ./src/

# Install dependencies and build
RUN pdm install --prod --no-editable
RUN pdm build

# Production stage
FROM python:3.13.7-slim as runtime

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Create non-root user
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid 1000 --create-home --shell /bin/bash appuser

# Install the package from wheel
COPY --from=builder /app/dist/*.whl /tmp/
RUN pip install --upgrade pip && \
    pip install /tmp/*.whl && \
    rm -rf /tmp/*.whl

# Create working directory for user
WORKDIR /data
RUN chown appuser:appuser /data

# Switch to non-root user
USER appuser

# Set default command
ENTRYPOINT ["milvus-ingest"]
CMD ["--help"]

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD milvus-ingest --help > /dev/null || exit 1

# Metadata
LABEL maintainer="zhuwenxing <wenxing.zhu@zilliz.com>" \
      description="Generate mock data for Milvus collections" \
      version="0.1.0" \
      org.opencontainers.image.source="https://github.com/zilliztech/milvus-ingest" \
      org.opencontainers.image.documentation="https://github.com/zilliztech/milvus-ingest/blob/main/README.md" \
      org.opencontainers.image.licenses="MIT"