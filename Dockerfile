FROM python:3.12-slim

# Install uv.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install the application dependencies.
WORKDIR /app

# Copy the application into the container.
COPY . /app

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 libgl1 -y

RUN uv sync --frozen --no-cache

# Expose port 80
EXPOSE 80

# Run the application.
CMD ["/app/.venv/bin/fastapi", "run", "main.py", "--port", "80", "--host", "0.0.0.0"]
