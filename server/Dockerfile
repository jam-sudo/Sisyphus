# Sisyphus engine API — self-contained image built from the exact checked-out tree.
#
# This Dockerfile works as-is for BOTH:
#   • a Hugging Face *Docker Space* — put this file at the Space repo root as
#     `Dockerfile`; HF builds it and serves on port 7860.
#   • a plain local build:  docker build -f server/Dockerfile -t sisyphus-api .
#                           docker run -p 7860:7860 sisyphus-api
#
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=7860

# system libs some wheels (rdkit / xgboost) expect at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
      libxrender1 libxext6 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml /app/
COPY src /app/src
COPY server/app.py server/config.py server/requirements.txt /app/server/
COPY data/model_card.json /app/data/model_card.json
COPY data/physiology /app/data/physiology
COPY data/enzymes /app/data/enzymes
COPY data/transporters /app/data/transporters
COPY data/sbi/prodrug_activation_registry.json /app/data/sbi/prodrug_activation_registry.json
COPY data/validation/development_residual_interval.json /app/data/validation/development_residual_interval.json
COPY data/validation/feature_schema_v1.json /app/data/validation/feature_schema_v1.json
COPY data/validation/solver_settings_v1.json /app/data/validation/solver_settings_v1.json
COPY models/adme /app/models/adme
COPY models/direct_pk /app/models/direct_pk

RUN pip install --requirement server/requirements.txt \
    && pip install -e . --no-deps

EXPOSE 7860
CMD ["sh", "-c", "uvicorn server.app:app --host 0.0.0.0 --port ${PORT:-7860} --workers 1"]
