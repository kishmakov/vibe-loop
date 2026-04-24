#!/usr/bin/env bash
set -euo pipefail

SRC_REPO="/tmp/trn"
DST_REPO="/home/kishmakov/Repos/ten-digits/data"

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <run_hash> [run_hash ...]"
  exit 1
fi

# echo "Init Aim repo at $DST_REPO if it doesn't exist..."
# uv run aim init --repo "$DST_REPO" >/dev/null 2>&1 || true

echo "Removing locks from runs in $SRC_REPO..."
timeout 30s uv run aim runs --repo "$SRC_REPO" close --yes "$@"
echo "Copying runs from $SRC_REPO to $DST_REPO..."
timeout 30s uv run aim runs --repo "$SRC_REPO" cp --destination "$DST_REPO" "$@"
