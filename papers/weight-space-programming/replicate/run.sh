#!/usr/bin/env bash
# One vLLM process per model so the GPU is freed between them.  Usage: ./run.sh <out name> <n> <samples> [models]
set -e; cd "$(dirname "$0")"
out=results/$1; n=${2:-100}; s=${3:-1}; models=${4:-unedited,p50c}
for m in ${models//,/ }; do uv run python probe.py --models $m --n $n --samples $s --out $out; done
