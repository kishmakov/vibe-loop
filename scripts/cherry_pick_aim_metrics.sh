#!/usr/bin/env bash
set -euo pipefail

SRC_REPO="/tmp/trn"
DST_REPO="/home/kishmakov/Repos/ten-digits/data"

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <run_hash> [run_hash ...]"
  exit 1
fi

# Create if doesn't exist
uv run aim init --repo "$DST_REPO" >/dev/null 2>&1 || true

# Runs written by a finished training job are left locked; close them first.
uv run aim runs --repo "$SRC_REPO" close --yes "$@"
uv run aim runs --repo "$SRC_REPO" cp --destination "$DST_REPO" "$@"
