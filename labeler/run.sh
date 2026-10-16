#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON="${LABELER_PYTHON:-/home/seungbo/anaconda3/envs/sam/bin/python}"
exec "$PYTHON" -m uvicorn app:app --host 127.0.0.1 --port "${PORT:-8000}" --workers 1
