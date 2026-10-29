#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export NO_ALBUMENTATIONS_UPDATE=1
exec python "${SCRIPT_DIR}/train.py" --config "${SCRIPT_DIR}/config.yaml" "$@"
