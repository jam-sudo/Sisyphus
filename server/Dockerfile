# Sisyphus engine API — self-contained image (clones the repo at build time).
#
# This Dockerfile works as-is for BOTH:
#   • a Hugging Face *Docker Space* — put this file at the Space repo root as
#     `Dockerfile`; HF builds it and serves on port 7860.
#   • a plain local build:  docker build -f server/Dockerfile -t sisyphus-api .
#                           docker run -p 7860:7860 sisyphus-api
#
# It pins to the repo's main HEAD at build time (rebuild the Space to update).
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=7860

# system libs some wheels (rdkit / xgboost) expect at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
      git libxrender1 libxext6 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# clone the engine (code + data + models) into /app
RUN git clone --depth 1 https://github.com/jam-sudo/Sisyphus.git /app
WORKDIR /app

RUN pip install -r requirements-lock.txt -r server/requirements.txt

EXPOSE 7860
CMD ["sh", "-c", "uvicorn server.app:app --host 0.0.0.0 --port ${PORT:-7860} --workers 1"]
