"""Test set demonstration for the 456-parameter 10-digit addition transformer.

Downloads the trained checkpoint from GCS (or uses a local path), then
evaluates on a diverse test set and prints human-readable results.

Usage:
    # Download checkpoint from GCS and evaluate:
    python demo_test.py

    # Use a local checkpoint:
    python demo_test.py --ckpt path/to/best.pt

    # Only run a quick smoke test (50 examples):
    python demo_test.py --quick

GCS checkpoint location (after training completes):
    gs://kishmakov-trans-count-outputs/456p_s43/checkpoints/best.pt
"""

import argparse
import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

from src.data import (
    MAX_OPERAND,
    PROMPT_LEN,
    SUM_DIGITS,
    TARGET_LEN,
    POW10_11,
    build_holdout_splits,
    postprocess,
    preprocess_batch,
)
from src.eval import evaluate_exact_match
from src.model_old import ModelConfig, TinyDecoderLM, count_parameters


GCS_CKPT = "gs://kishmakov-trans-count-outputs/456p_s43/checkpoints/best.pt"

# Seeds for multi-seed evaluation (same as evaluate_checkpoints.py)
EVAL_SEEDS = [41, 100, 200, 300, 400, 500, 999, 1234, 7777, 31415]
TEST_SIZE = 10_000
VAL_SIZE = 5_000


# ---------------------------------------------------------------------------
# Hand-picked demonstration examples covering easy → hard cases
# ---------------------------------------------------------------------------
DEMO_EXAMPLES = [
    # Simple
    (0, 0),
    (1, 1),
    (5, 7),
    (42, 58),
    # Medium
    (12345, 67890),
    (999999, 1),
    (500000, 500000),
    # Hard (10-digit)
    (9999999999, 1),
    (1234567890, 9876543210),
    (5000000000, 5000000000),
    (3141592653, 2718281828),
    (9999999999, 9999999999),
]


def download_from_gcs(gcs_path: str, local_path: Path) -> bool:
    """Download a file from GCS. Returns True on success."""
    try:
        result = subprocess.run(
            ["gsutil", "cp", gcs_path, str(local_path)],
            capture_output=True, text=True, timeout=120,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def load_checkpoint(ckpt_path: Path, device: str) -> TinyDecoderLM:
    blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ModelConfig(**blob["model_config"])
    model = TinyDecoderLM(cfg).to(device)
    model.load_state_dict(blob["model_state"])
    model.eval()
    return model, blob


@torch.no_grad()
def predict_batch(
    model: TinyDecoderLM, pairs: list[tuple[int, int]], device: str
) -> list[dict]:
    """Run inference on a list of (a, b) pairs and return prediction dicts."""
    a = torch.tensor([p[0] for p in pairs], dtype=torch.int64)
    b = torch.tensor([p[1] for p in pairs], dtype=torch.int64)
    prompt = preprocess_batch(a, b).to(device)
    gen = model.generate(prompt, max_new_tokens=TARGET_LEN)
    pred_tail = gen[:, -TARGET_LEN:].cpu()

    results = []
    for i, (ai, bi) in enumerate(pairs):
        pred = postprocess(pred_tail[i].tolist())
        gt = ai + bi
        results.append({
            "A": ai, "B": bi,
            "prediction": pred,
            "ground_truth": gt,
            "correct": pred == gt,
        })
    return results


def print_demo_predictions(model: TinyDecoderLM, device: str) -> None:
    print("\n" + "=" * 70)
    print("DEMONSTRATION EXAMPLES")
    print("=" * 70)
    print(f"{'A':>20}  {'B':>20}  {'Prediction':>21}  OK?")
    print("-" * 70)

    results = predict_batch(model, DEMO_EXAMPLES, device)
    n_correct = sum(r["correct"] for r in results)

    for r in results:
        status = "✓" if r["correct"] else "✗"
        print(f"{r['A']:>20}  {r['B']:>20}  {r['prediction']:>21}  {status}")

    print("-" * 70)
    print(f"Demonstration accuracy: {n_correct}/{len(DEMO_EXAMPLES)}")


def run_quick_test(model: TinyDecoderLM, device: str, n: int = 50) -> None:
    """Smoke test on random 10-digit operands."""
    print(f"\n{'=' * 70}")
    print(f"QUICK TEST ({n} random 10-digit examples)")
    print("=" * 70)
    rng = random.Random(42)
    pairs = [(rng.randint(0, MAX_OPERAND - 1), rng.randint(0, MAX_OPERAND - 1))
             for _ in range(n)]
    results = predict_batch(model, pairs, device)
    n_correct = sum(r["correct"] for r in results)
    failures = [r for r in results if not r["correct"]]
    print(f"Correct: {n_correct}/{n}  ({100 * n_correct / n:.1f}%)")
    if failures:
        print(f"\nFirst {min(3, len(failures))} failure(s):")
        for r in failures[:3]:
            print(f"  {r['A']} + {r['B']} = {r['prediction']} (expected {r['ground_truth']})")


def run_full_evaluation(
    model: TinyDecoderLM, device: str, batch_size: int, data_dir: Path
) -> dict:
    print(f"\n{'=' * 70}")
    print(f"FULL EVALUATION ({len(EVAL_SEEDS)} seeds × {TEST_SIZE:,} examples"
          f" = {len(EVAL_SEEDS) * TEST_SIZE:,} total)")
    print("=" * 70)

    data_dir.mkdir(parents=True, exist_ok=True)
    per_seed = {}
    total_errors = 0

    for seed in EVAL_SEEDS:
        split_path = data_dir / f"holdout_v{VAL_SIZE}_t{TEST_SIZE}_seed{seed}.pt"
        splits = build_holdout_splits(VAL_SIZE, TEST_SIZE, seed, split_path)
        em, _ = evaluate_exact_match(
            model, splits["test_a"], splits["test_b"], batch_size, device
        )
        errors = int(round((1 - em) * TEST_SIZE))
        total_errors += errors
        per_seed[seed] = {"exact_match": em, "errors": errors}
        status = "PASS" if errors == 0 else f"FAIL ({errors} errors)"
        print(f"  seed={seed:>5}: exact_match={em:.6f}  {status}")

    total = len(EVAL_SEEDS) * TEST_SIZE
    aggregate = 1 - total_errors / total
    print(f"\nAggregate: {aggregate:.6f} ({total_errors} errors in {total:,} examples)")
    return {"aggregate_exact_match": aggregate, "total_errors": total_errors,
            "total_examples": total, "per_seed": per_seed}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Demo & test the 456-parameter 10-digit addition transformer"
    )
    parser.add_argument(
        "--ckpt", type=Path, default=None,
        help="Path to checkpoint .pt file (downloads from GCS if not provided)",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--data-dir", type=Path, default=Path("results/data"))
    parser.add_argument(
        "--quick", action="store_true",
        help="Run a quick smoke test (50 examples) instead of full evaluation",
    )
    parser.add_argument("--out", type=Path, default=None,
                        help="Save full results to this JSON file")
    args = parser.parse_args()

    # --- Resolve checkpoint ---
    ckpt_path = args.ckpt
    tmp_dir = None
    if ckpt_path is None:
        print(f"No checkpoint specified. Attempting download from GCS:")
        print(f"  {GCS_CKPT}")
        tmp_dir = tempfile.mkdtemp()
        ckpt_path = Path(tmp_dir) / "best.pt"
        if not download_from_gcs(GCS_CKPT, ckpt_path):
            print("\nDownload failed. Either:")
            print("  1. Training is still in progress (check Vertex AI job status)")
            print("  2. Pass --ckpt path/to/best.pt for a local checkpoint")
            print("\nTo check job status:")
            print("  gcloud ai custom-jobs list --region=us-central1")
            sys.exit(1)
        print("Download successful.\n")

    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    # --- Load model ---
    print(f"Loading checkpoint: {ckpt_path}")
    model, blob = load_checkpoint(ckpt_path, args.device)
    params = count_parameters(model)
    train_step = blob.get("step", "unknown")
    val_exact = blob.get("val_exact", "unknown")
    print(f"  Parameters : {params}")
    print(f"  Config     : {model.cfg}")
    print(f"  Checkpoint : step={train_step}, best_val_exact={val_exact}")

    # --- Demo predictions ---
    print_demo_predictions(model, args.device)

    # --- Evaluation ---
    if args.quick:
        run_quick_test(model, args.device)
        return

    eval_results = run_full_evaluation(model, args.device, args.batch_size, args.data_dir)

    # --- Save results ---
    if args.out:
        out_path = args.out
    else:
        out_path = Path(f"results/demo_test_{params}params.json")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    full_results = {
        "checkpoint": str(ckpt_path),
        "parameters": params,
        "model_config": {
            k: getattr(model.cfg, k)
            for k in ["n_layer", "d_model", "n_head", "d_ff",
                      "pos_rank", "qkv_rank", "attn_out_rank", "ffn_rank",
                      "use_rmsnorm", "tie_qkv"]
        },
        "training_step": train_step,
        "best_val_exact": val_exact,
        "evaluation": eval_results,
    }
    with open(out_path, "w") as f:
        json.dump(full_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
