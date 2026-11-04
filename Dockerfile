FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/var/cache/babyinfoquick
WORKDIR /app
RUN useradd --create-home --uid 10001 biq \
    && mkdir -p /app/data "$HF_HOME" \
    && chown biq:biq /app/data "$HF_HOME"
COPY pyproject.toml ./
RUN python -c "import tomllib; print('\n'.join(tomllib.load(open('pyproject.toml','rb'))['project']['dependencies']))" > /tmp/app-requirements.txt \
    && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch \
    && pip install --no-cache-dir -r /tmp/app-requirements.txt
COPY babyinfoquick ./babyinfoquick
RUN pip install --no-cache-dir --no-deps -e .
COPY configs ./configs
COPY templates ./templates
COPY eval ./eval
COPY deploy ./deploy
USER biq
EXPOSE 8000
CMD ["uvicorn", "babyinfoquick.api:app", "--host", "0.0.0.0", "--port", "8000", "--limit-concurrency", "8", "--no-access-log"]
