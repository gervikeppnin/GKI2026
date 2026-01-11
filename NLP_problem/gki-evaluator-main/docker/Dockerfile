FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app
RUN mkdir -p /app/submission /tmp/submission

# Install ML frameworks (CPU-only) and common libraries
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir \
    transformers \
    safetensors \
    tensorflow-cpu \
    jax \
    flax \
    numpy \
    scipy

# Non-root user for security
RUN adduser --disabled-password --gecos "" competitor \
    && chown -R competitor:competitor /app /tmp/submission

COPY scoring.py /app/scoring.py
RUN chmod 644 /app/scoring.py

USER competitor
ENTRYPOINT ["python", "scoring.py"]