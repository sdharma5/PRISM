FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip install --no-cache-dir -e ".[api,recommendations,speech-realtime,documents]"

EXPOSE 8080

CMD ["uvicorn", "apps.api.main:app", "--host", "0.0.0.0", "--port", "8080"]
