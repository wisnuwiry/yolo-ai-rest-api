FROM python:3.12-slim

# Install uv.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Copy the application into the container
COPY . /app

# Install system dependencies
RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 libgl1

# Sync dependencies with uv
RUN uv sync --frozen --no-cache

# Expose port 80
EXPOSE 80

# Run the application using uvicorn
CMD ["/app/.venv/bin/uvicorn", "main:app", "--port", "80", "--host", "0.0.0.0"]
