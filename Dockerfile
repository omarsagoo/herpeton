FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System deps for Pillow/torchvision image backends
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libjpeg-dev \
        zlib1g-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install PyTorch/torchvision from the CPU wheel index, then the rest of the deps
RUN python -m pip install --no-cache-dir --upgrade pip
RUN python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
    torch==2.2.2 torchvision==0.17.2

COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

# Copy application code (includes model files under app/models)
COPY app ./app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
