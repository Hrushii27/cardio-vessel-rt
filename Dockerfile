# Multi-stage Dockerfile for CardioRT
FROM python:3.11-slim

# Prevent Python from writing pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system libraries needed by OpenCV and GUI
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .
RUN pip install --no-cache-dir -e . --no-deps

# Default port for Streamlit web app
EXPOSE 8501

# Default command: run benchmark suite to certify latency
CMD ["python", "scripts/benchmark.py", "--frames", "500"]
