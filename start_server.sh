#!/usr/bin/env bash
# Start FunASR SenseVoice HTTP server using the local .venv
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

if [[ ! -d .venv ]]; then
  echo "No .venv found. Create it with:"
  echo "  python3 -m venv .venv"
  echo "  source .venv/bin/activate"
  echo "  pip install -r requirements.txt"
  exit 1
fi

# shellcheck disable=SC1091
source .venv/bin/activate

export PYTHONUNBUFFERED=1
# Prefer HuggingFace/ModelScope caches under the project if desired:
# export MODELSCOPE_CACHE="$ROOT/.cache/modelscope"
# export HF_HOME="$ROOT/.cache/huggingface"

exec python server.py --model sensevoice --device mps "$@"
