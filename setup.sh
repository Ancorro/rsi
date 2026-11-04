#!/usr/bin/env bash
# CPU-only setup: .venv/ with CPU torch, transformers, pytest and ruff. Then: bash check.sh
# Needs network access for pip (PyPI and the PyTorch CPU index). Real model runs need a GPU; see README.md.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python3.12}
command -v "$PY" >/dev/null || PY=python3
"$PY" -c 'import sys; assert sys.version_info[:2] >= (3, 11), "Python 3.11+ needed"'
[ -d .venv ] || "$PY" -m venv .venv
.venv/bin/python -m pip install -q --upgrade pip
.venv/bin/python -m pip install -q --index-url https://download.pytorch.org/whl/cpu "torch==2.13.0"
.venv/bin/python -m pip install -q -e . "pytest>=8.3" "ruff>=0.9"
.venv/bin/python -c 'import torch, transformers; print(f"ok: torch {torch.__version__}, transformers {transformers.__version__}")'
