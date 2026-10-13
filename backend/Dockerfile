# backend/Dockerfile
FROM ultralytics/ultralytics:latest-jetson

# The Ultralytics image comes with CUDA, OpenCV, and PyTorch.
# We just need to ensure the working directory is set.
RUN apt-get update && apt-get install -y \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download YOLO model to save time on the Pi
# Removed: The file is already included locally, downloading it hangs the build.

# Copy application code
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Run uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

