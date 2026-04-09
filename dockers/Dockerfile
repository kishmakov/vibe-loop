FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

WORKDIR /app

ENV PYTHONUNBUFFERED=1

# Install Python dependencies
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu121 || true

# Install gsutil (part of google-cloud-sdk) for GCS upload
RUN apt-get update -qq && apt-get install -y -qq curl gnupg && \
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" \
      | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
      | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg && \
    apt-get update -qq && apt-get install -y -qq google-cloud-cli && \
    rm -rf /var/lib/apt/lists/*

# Copy source code
COPY src/ /app/src/
COPY evaluate_checkpoints.py /app/evaluate_checkpoints.py
COPY scripts/train_and_upload.sh /app/train_and_upload.sh
RUN chmod +x /app/train_and_upload.sh

# Create output directories
RUN mkdir -p /app/results/runs /app/results/data /app/checkpoints

# Default: train the 456-parameter model (seed 43), upload checkpoints to GCS.
# Reference: https://github.com/yinglunz/A-456-Parameter-Transformer-Solves-10-Digit-Addition
#   pos_rank=3, qkv_rank=3, attn_out_rank=2, ffn_rank=3
#   tie_qkv=shareA_tieKV, use_rmsnorm=True
#   Total: 98 (token_emb) + 120 (pos_emb) + 7 (ln1) + 63 (qkv)
#          + 28 (attn_out) + 7 (ln2) + 126 (mlp) + 7 (ln_f) = 456 params
CMD ["/app/train_and_upload.sh"]
