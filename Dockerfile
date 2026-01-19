# Dockerfile for Video Classification System
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories for data
RUN mkdir -p data/raw_videos data/tamper data/mlruns

# Expose ports
EXPOSE 5000 8501 5000

# Health check
HEALTHCHECK CMD python -c "import requests; requests.get('http://localhost:5000/health')" || exit 1

# Default command
CMD ["python", "app/app.py"]