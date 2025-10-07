FROM python:3.11-slim

WORKDIR /app

# Minimal environment variables
ENV HF_HOME=/tmp/huggingface
ENV TRANSFORMERS_CACHE=/tmp/huggingface
ENV SENTENCE_TRANSFORMERS_HOME=/tmp/huggingface
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# Install minimal system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Copy and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    rm -rf /root/.cache /tmp/*

# Copy application
COPY main.py .env* ./

# Don't pre-download models - let them load on first request
# This saves ~2GB in image size

EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860", "--log-level", "warning"] 