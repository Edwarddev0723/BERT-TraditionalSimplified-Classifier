#!/usr/bin/env bash
set -euo pipefail

python -m bert_ts_classifier.training.train --config configs/train.yaml "$@"
