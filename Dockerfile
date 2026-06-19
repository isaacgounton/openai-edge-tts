FROM python:3.12-slim

WORKDIR /app

# ffmpeg is required: the streaming PCM path pipes edge-tts mp3 -> s16le PCM
# through ffmpeg, and the wav/opus/aac convert paths use it too. curl powers the
# container HEALTHCHECK.
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install them
COPY requirements.txt /app
RUN pip install -r requirements.txt

# Copy the app directory
COPY app/ /app

EXPOSE 5050

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${PORT:-5050}/health" || exit 1

# Command to run the server
CMD ["python", "/app/server.py"]
