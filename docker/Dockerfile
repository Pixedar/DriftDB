FROM python:3.11-slim

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir -e .

ENV DRIFTDB_CHROMA_PATH=/data/driftdb_chroma
EXPOSE 8000

CMD ["driftdb"]
