# Use an official Python base image that supports TensorFlow 2.15.0
FROM python:3.10-slim

# Set environment variables to prevent Python from writing pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Create and set the working directory
WORKDIR /app

# Install OS-level dependencies required for DeepFace and other packages
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (leveraging Docker layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Expose the port your app runs on
EXPOSE 8001

# Start the FastAPI app using uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
