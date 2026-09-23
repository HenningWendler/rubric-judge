FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app
COPY pyproject.toml constraints.txt README.md ./
COPY src ./src
RUN pip install --no-cache-dir --constraint constraints.txt .

# A numeric user, because Kubernetes' runAsNonRoot can only verify a UID, not a name.
RUN useradd --system --uid 10001 rubric-judge
USER 10001

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --start-interval=1s \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"]

# One process per container: RUBRIC_JUDGE_MAX_CONCURRENT is a per-process limit, so
# uvicorn workers would multiply it. Scale by running more containers.
CMD ["uvicorn", "rubric_judge.api:app", "--host", "0.0.0.0", "--port", "8000"]
