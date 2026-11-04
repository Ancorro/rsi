#!/usr/bin/env bash
# CPU checks, under a minute: unit tests on a tiny random Qwen3.5, then ruff.
set -euo pipefail
cd "$(dirname "$0")"
.venv/bin/python -m pytest -q
.venv/bin/python -m ruff check .
