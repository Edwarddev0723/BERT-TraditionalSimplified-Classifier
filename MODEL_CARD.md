# Model Card: BERT Traditional Chinese Classifier

- Task: Text classification — Mainland Traditional vs Taiwan Traditional Chinese
- Version: v0.1.0
- License: Apache-2.0

## Overview
This repository modularizes a previous notebook-only workflow into a reproducible package.
It trains or serves a BERT-based classifier for distinguishing two Traditional Chinese styles.

## Data
- Synthetic sample provided via `scripts/preprocess.py` for E2E runs.
- If using private data, document the schema as:
  - `text`: string
  - `label`: int (0=Mainland Trad, 1=Taiwan Trad)

### Limitations
- Labels may overlap for mixed/ambiguous vocabulary.
- Long documents truncated at `max_len` (configurable).

## Training Details
- Backbone: `ckiplab/bert-base-chinese`
- Loss: CrossEntropy (optionally FocalLoss if available)
- Optimizer: AdamW
- Max epochs, LR, batch size configurable via YAML overrides.

## Metrics
We report standard classification metrics:
- Accuracy
- F1 (macro)
- AUPRC (binary)

Run evaluation to regenerate metrics and confusion matrix:

```
python -m bert_ts_classifier.evaluation.eval --config configs/eval.yaml
```

Artifacts written to `runs/eval/`.

## Risks and Ethical Considerations
- The classifier detects stylistic differences in Chinese writing and must not be used for demographic profiling.
- Domain shift and slang may reduce performance.

## Intended Use
- Content analysis, preprocessing, or routing.
- Not intended for sensitive decision making.

## Changelog
See `CHANGELOG.md`.
