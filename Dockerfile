# Use the official Python slim image for a smaller footprint
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

# Set the working directory
WORKDIR /app

# Install system dependencies required for building some packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file or pyproject.toml first to leverage Docker cache
COPY pyproject.toml /app/

# Install the application
RUN pip install --no-cache-dir -e .

# Copy the rest of the application code
COPY . /app/

# Expose the port the app runs on
EXPOSE 8000

# Start the FastAPI application via uvicorn
CMD ["python", "-m", "app", "--host", "0.0.0.0", "--port", "8000"]
