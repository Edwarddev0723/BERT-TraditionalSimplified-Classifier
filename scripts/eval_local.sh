#!/usr/bin/env bash
set -euo pipefail

python -m bert_ts_classifier.evaluation.eval --config configs/eval.yaml "$@"
