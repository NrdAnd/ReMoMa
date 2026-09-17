FROM python:3.10-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m pip install --no-cache-dir torch==2.2.0 --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install --no-cache-dir .
COPY scripts ./scripts
COPY configs ./configs
COPY data/graphs ./data/graphs
RUN useradd --create-home --uid 10001 remoma && mkdir -p /app/data/raw /app/data/processed /app/runs \
    && chown -R remoma:remoma /app
USER remoma
CMD ["python", "scripts/train.py", "--help"]
