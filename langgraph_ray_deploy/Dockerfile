# Base image is expected to be pulled locally (python:3.14-slim) -- the
# build must not need registry access. Only PyPI is hit, inside the build.
FROM python:3.14-slim

WORKDIR /app

# Lockfile first for layer caching, then the package itself.
COPY pyproject.toml uv.lock README.md ./
COPY src/ ./src/
RUN pip install --no-cache-dir .

EXPOSE 8000

# serve.run(..., blocking=True) in main.py keeps this process -- and the
# Ray cluster it spawns as children -- alive for the life of the container.
# If the app dies, the container exits and docker restarts it.
ENTRYPOINT ["python", "-m", "langgraph_ray_deploy.main"]
