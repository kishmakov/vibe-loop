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
- [x] `dockers/Dockerfile` — Docker container for Vertex AI training

### In Progress
- [ ] Submit training job to Vertex AI

### Pending
- [ ] Monitor training (expect grokking ~40K-54K steps for seed 43)
- [ ] Verify 456-parameter count after training
- [ ] Evaluate checkpoint on 100K test examples

## 456-Parameter Configuration

Derived by solving: total_params = 456

| Component | Config | Params |
|-----------|--------|--------|
| token_emb | 14 × 7, tied with lm_head | 98 |
| pos_emb | LowRankEmbedding(33, 7, rank=3) | 120 |
| ln1 | RMSNorm(7) | 7 |
| QKV (shareA_tieKV) | A(7×5) + Bq(5×7) + Bkv(5×7) | 105 |
| attn_out | LowRankLinear(7, 7, rank=2) | 28 |
| ln2 | RMSNorm(7) | 7 |
| MLP fc1 | LowRankLinear(7, 14, rank=2) | 42 |
| MLP fc2 | LowRankLinear(14, 7, rank=2) | 42 |
| ln_f | RMSNorm(7) | 7 |
| **Total** | | **456** |

Training command:
```bash
python -m src.train \
  --run-name 456p_s43 \
  --pos-rank 3 --qkv-rank 5 --attn-out-rank 2 --ffn-rank 2 \
  --tie-qkv shareA_tieKV --use-rmsnorm \
  --seed 43 --train-steps 54000 --warmup-steps 2700 \
  --device cuda --dtype bf16
```

## Notes
- Grokking phenomenon: accuracy stays near 0 for ~40K steps then jumps to ~100%
- Only 2 of 5 seeds succeed within 54K steps (seeds 43 and 44 in original paper)
- Paper used seed 43 for the 456-param model with 100% exact-match
