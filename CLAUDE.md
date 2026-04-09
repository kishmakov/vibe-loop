# VM

To access Google Cloud VM, use this command: `gcloud compute ssh instance-20260409-080533 --zone=us-central1-c`

# Google Cloud

- **Project:** `project-39b7a321-7b37-42c3-bdb`
- **Region:** `us-central1`
- **Artifact Registry repo:** `kishmakov-dockers`

# Docker

**Local disk is full — always use Cloud Build, never `scripts/build_docker.sh`.**

Build and push via Cloud Build:

```bash
gcloud builds submit --config cloudbuild.yaml .
```

Image URI: `us-central1-docker.pkg.dev/project-39b7a321-7b37-42c3-bdb/kishmakov-dockers/trans-count:latest`

Requires `dockers/Dockerfile` at the repo root.

# Training

Submit job to Google Cloud Vertex AI:

```bash
scripts/submit_training.sh
```

Training outputs (checkpoints, metrics, config) are uploaded to GCS on completion:
`gs://kishmakov-trans-count-outputs/<RUN_NAME>/`

# Results

The 456-parameter model was successfully trained (seed 43, fp32):
- **100% exact match** on 10-digit addition at step 34,000 (matches reference paper)
- Checkpoint: `gs://kishmakov-trans-count-outputs/456p_s43/checkpoints/best.pt`

To evaluate with human-readable predictions (requires PyTorch):
```bash
# Quick smoke test (50 examples)
python3 demo_test.py --quick

# Full evaluation (10 seeds × 10K examples)
python3 demo_test.py
```

Run from the VM or inside the container (`scripts/run_demo_test.sh` downloads checkpoint from GCS automatically).
