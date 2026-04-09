#!/usr/bin/env bash
# Runs training then uploads all output artifacts to GCS.
# Designed to run inside the Vertex AI container.
# Usage: train_and_upload.sh [GCS_BUCKET] [extra python args...]
set -e

GCS_BUCKET="${GCS_BUCKET:-gs://kishmakov-trans-count-outputs}"
RUN_NAME="${RUN_NAME:-456p_s43}"
RUN_DIR="/app/results/runs/${RUN_NAME}"
GCS_OUTPUT="${GCS_BUCKET}/${RUN_NAME}"

echo "=== Training ==="
python -m src.train \
  --run-name "${RUN_NAME}" \
  --pos-rank 3 --qkv-rank 3 --attn-out-rank 2 --ffn-rank 3 \
  --tie-qkv shareA_tieKV --use-rmsnorm \
  --seed 43 --train-steps 54000 --warmup-steps 1350 \
  --device cuda --dtype fp32 \
  --run-dir "${RUN_DIR}"

echo "=== Uploading to ${GCS_OUTPUT} ==="
gsutil -m cp -r "${RUN_DIR}" "${GCS_OUTPUT}"

echo "=== Done. Artifacts at ${GCS_OUTPUT} ==="
