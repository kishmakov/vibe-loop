# Agent TODO

## Goal
Reproduce https://github.com/yinglunz/A-456-Parameter-Transformer-Solves-10-Digit-Addition

A 456-parameter transformer that achieves 100% exact-match accuracy on 10-digit addition.

## Status

### Completed
- [x] `src/model.py` — Low-rank transformer architecture (LowRankLinear, LowRankEmbedding, RMSNorm, CausalSelfAttention with 10 QKV tying modes, MLP, Block, TinyDecoderLM)
- [x] `src/data.py` — Data pipeline (tokenization, vectorized encoding, curriculum batch sampling, holdout splits)
- [x] `src/train.py` — Training with 3-phase curriculum learning, cosine LR, AdamW
- [x] `src/eval.py` — Evaluation and inference
- [x] `evaluate_checkpoints.py` — Multi-seed checkpoint evaluation
- [x] `dockers/Dockerfile` — Docker container with gsutil for GCS checkpoint upload
- [x] `scripts/train_and_upload.sh` — Training wrapper that uploads artifacts to GCS on completion
- [x] `cloudbuild.yaml` — Cloud Build config for building Docker image remotely
- [x] `demo_test.py` — Test set demonstration script (downloads checkpoint from GCS, evaluates on 100K examples across 10 seeds, shows human-readable predictions)
- [x] GCS bucket `gs://kishmakov-trans-count-outputs/` created for training outputs
- [x] Docker image rebuilt with GCS upload support (build ID: 7f3de47a)
- [x] Training job submitted to Vertex AI (job ID: 1312207049748119552)

### In Progress
- [ ] Monitor fp32 training job (job ID: 4417491769378209792)
  - **Key fix**: reference used fp32 dtype (not bf16 as we did initially)
  - Reference achieved 100% exact match at step ~34000 with fp32 + seed 43
  - First run (bf16): 54K steps, token acc 39%, exact match 0% (no grokking)
  - Check logs: `gcloud ai custom-jobs stream-logs projects/275442587350/locations/us-central1/customJobs/4417491769378209792`
  - Outputs: `gs://kishmakov-trans-count-outputs/456p_s43/` (will be overwritten)

### Pending
- [ ] Run `demo_test.py` once checkpoint appears in GCS:
  ```bash
  python demo_test.py  # auto-downloads from GCS and runs full evaluation
  ```

## 456-Parameter Configuration (reference-exact)

Matches https://github.com/yinglunz/A-456-Parameter-Transformer-Solves-10-Digit-Addition exactly.

| Component | Config | Params |
|-----------|--------|--------|
| token_emb | 14 × 7, tied with lm_head | 98 |
| pos_emb | LowRankEmbedding(33, 7, rank=3) | 120 |
| ln1 | RMSNorm(7) | 7 |
| QKV (shareA_tieKV) | A(7×3) + Bq(3×7) + Bkv(3×7) | 63 |
| attn_out | LowRankLinear(7, 7, rank=2) | 28 |
| ln2 | RMSNorm(7) | 7 |
| MLP fc1 | LowRankLinear(7, 14, rank=3) | 63 |
| MLP fc2 | LowRankLinear(14, 7, rank=3) | 63 |
| ln_f | RMSNorm(7) | 7 |
| **Total** | | **456** |

Training command (inside container):
```bash
python -m src.train \
  --run-name 456p_s43 \
  --pos-rank 3 --qkv-rank 3 --attn-out-rank 2 --ffn-rank 3 \
  --tie-qkv shareA_tieKV --use-rmsnorm \
  --seed 43 --train-steps 54000 --warmup-steps 1350 \
  --device cuda --dtype bf16
```

## Notes
- Grokking phenomenon: accuracy stays near 0 for ~40K steps then jumps to ~100%
- Only 2 of 5 seeds succeed within 54K steps (seeds 43 and 44 in original paper)
- Paper used seed 43 for the 456-param model with 100% exact-match
- Cloud Build API was enabled and compute SA granted artifactregistry.writer role
- Local disk is full; always use `gcloud builds submit` instead of `scripts/build_docker.sh`
- GCS bucket `gs://kishmakov-trans-count-outputs/` in us-central1 stores training outputs

## Evaluation (once checkpoint available)

```bash
# Full evaluation: 10 seeds × 10K examples = 100K total
python demo_test.py

# Quick smoke test: 50 random examples  
python demo_test.py --quick

# With local checkpoint:
python demo_test.py --ckpt results/runs/456p_s43/checkpoints/best.pt
```
