# VM

To access Google Cloud VM, use this command: `gcloud compute ssh instance-20260409-080533 --zone=us-central1-c`

# Google Cloud

- **Project:** `project-39b7a321-7b37-42c3-bdb`
- **Region:** `us-central1`
- **Artifact Registry repo:** `kishmakov-dockers`

# Docker

Build and push the `trans-count` image:

```bash
scripts/build_docker.sh
```

Image URI: `us-central1-docker.pkg.dev/project-39b7a321-7b37-42c3-bdb/kishmakov-dockers/trans-count:latest`

Override the tag: `IMAGE_TAG=v1.0 scripts/build_docker.sh`

Requires `dockers/Dockerfile` at the repo root.

# Training

Submit job to Google Cloud Vertex AI:

```bash
scripts/submit_training.sh
```
