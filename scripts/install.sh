#!/usr/bin/env bash
# Installs dependencies with the CPU-only torch wheel (~200 MB instead of ~4 GB).
set -euo pipefail
cd "$(dirname "$0")/.."

python3 -m pip install --upgrade pip
python3 -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python3 -m pip install -r requirements.txt

python3 - <<'PY'
import torch
print(f"torch {torch.__version__}  cuda_build={'+cu' in torch.__version__}")
PY
