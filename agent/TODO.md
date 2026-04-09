# Agent TODO

## Goal
Reproduce https://github.com/yinglunz/A-456-Parameter-Transformer-Solves-10-Digit-Addition

A 456-parameter transformer that achieves 100% exact-match accuracy on 10-digit addition.

## Status: COMPLETE ✓

### Completed
- [x] `src/model.py` — Low-rank transformer architecture (LowRankLinear, LowRankEmbedding, RMSNorm, CausalSelfAttention with 10 QKV tying modes, MLP, Block, TinyDecoderLM)
- [x] `src/data.py` — Data pipeline (tokenization, vectorized encoding, curriculum batch sampling, holdout splits)
- [x] `src/train.py` — Training with 3-phase curriculum learning, cosine LR, AdamW
- [x] `src/eval.py` — Evaluation and inference
- [x] `evaluate_checkpoints.py` — Multi-seed checkpoint evaluation
- [x] `dockers/Dockerfile` — Docker container with gsutil for GCS checkpoint upload
- [x] `scripts/train_and_upload.sh` — Training wrapper that uploads artifacts to GCS on completion
- [x] `scripts/run_demo_test.sh` — Post-training evaluation script (downloads checkpoint from GCS, runs demo_test.py)
- [x] `cloudbuild.yaml` — Cloud Build config for building Docker image remotely
- [x] `demo_test.py` — Test set demonstration script (downloads checkpoint from GCS, evaluates on 100K examples across 10 seeds, shows human-readable predictions)
- [x] GCS bucket `gs://kishmakov-trans-count-outputs/` created for training outputs
- [x] Training job completed — **100% exact match at step 34,000** (matches reference paper exactly)
- [x] Checkpoint saved to `gs://kishmakov-trans-count-outputs/456p_s43/checkpoints/best.pt`

## Results

| Metric | Value |
|--------|-------|
| Exact match accuracy | **100%** |
| Grokking step | 34,000 |
| Total train steps | 54,000 |
| Training time | ~20 min |
| Parameters | 456 |
| Seed | 43 |

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

## Notes
- Grokking phenomenon: accuracy stays near 0 for ~34K steps then jumps to 100%
- fp32 dtype is required — bf16 failed to grok (0% exact match after 54K steps)
- Local disk is full; always use `gcloud builds submit` instead of `scripts/build_docker.sh`
- GCS bucket `gs://kishmakov-trans-count-outputs/` in us-central1 stores training outputs
