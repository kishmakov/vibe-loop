#!/usr/bin/env bash

REGION="us-central1"
IMAGE_NAME="trans-count"
IMAGE_TAG="${IMAGE_TAG:-latest}"

IMAGE_URI="${REGION}-docker.pkg.dev/project-39b7a321-7b37-42c3-bdb/kishmakov-dockers/${IMAGE_NAME}:${IMAGE_TAG}"

gcloud ai custom-jobs create \
  --region="${REGION}" \
  --display-name=${IMAGE_NAME} \
  --worker-pool-spec=machine-type=n1-standard-4,replica-count=1,accelerator-type=NVIDIA_TESLA_T4,accelerator-count=1,container-image-uri="${IMAGE_URI}"
