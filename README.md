# BERT Traditional Chinese Classifier

Modular, installable, and testable project for classifying Traditional Chinese text into two styles: Mainland vs Taiwan. Converted from notebooks to a reproducible Python package.

- Package name: `bert_ts_classifier`
- Python: 3.10
- License: Apache-2.0

## Quick Start

1) Install and set up environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
pre-commit install
```

2) Prepare a tiny synthetic dataset (safe to share)

```bash
python scripts/preprocess.py
```

3) Train and evaluate

```bash
python -m bert_ts_classifier.training.train --config configs/train.yaml
python -m bert_ts_classifier.evaluation.eval --config configs/eval.yaml
```

Override any config on the fly:

```bash
python -m bert_ts_classifier.training.train --config configs/train.yaml \
  train.max_epochs=3 data.max_len=192 data.batch_size=4
```

## Data Format

Input files are JSONL with one object per line:

```json
{"text": "這個軟體的介面設計得很好。", "label": 1}
```

- label: 0 = Mainland Traditional, 1 = Taiwan Traditional
- See `scripts/preprocess.py` for synthetic data generation.

## Project Layout

```
.
├─ README.md
├─ LICENSE
├─ MODEL_CARD.md
├─ CHANGELOG.md
├─ pyproject.toml
├─ .gitignore
├─ .pre-commit-config.yaml
├─ .github/workflows/
│  ├─ quality.yml
│  └─ tests.yml
├─ configs/
│  ├─ data.yaml
│  ├─ model.yaml
│  ├─ train.yaml
│  └─ eval.yaml
├─ src/bert_ts_classifier/
│  ├─ __init__.py
│  ├─ data/
│  │  └─ datamodule.py
│  ├─ models/
│  │  └─ bert_classifier.py
│  ├─ training/
│  │  └─ train.py
│  ├─ evaluation/
│  │  └─ eval.py
│  ├─ infer/
│  │  ├─ predict.py
│  │  └─ serve_fastapi.py
│  ├─ export/
│  │  ├─ to_onnx.py
│  │  └─ to_torchscript.py
│  └─ utils/
│     ├─ seed.py
│     ├─ logging.py
│     └─ io.py
├─ scripts/
│  ├─ convert_ipynb.py
│  ├─ preprocess.py
│  ├─ train_local.sh
│  └─ eval_local.sh
├─ tests/
│  ├─ test_data.py
│  ├─ test_model.py
│  └─ test_metrics.py
├─ notebooks/
└─ examples/
```

## Reproducibility

- All hyper-parameters live in `configs/*.yaml`.
- Deterministic seeds via `bert_ts_classifier.utils.seed`.
- CI validates lint + type + tests on PRs and pushes.

## Inference and Serving

- Batch inference: `python -m bert_ts_classifier.infer.predict --config configs/eval.yaml --input samples.csv --output preds.csv`
- FastAPI service: `uvicorn bert_ts_classifier.infer.serve_fastapi:app --host 0.0.0.0 --port 8000`
  - /health and /predict endpoints

## Export

- ONNX: `python -m bert_ts_classifier.export.to_onnx`
- TorchScript: `python -m bert_ts_classifier.export.to_torchscript`

## From Notebook to Modules

- Use `scripts/convert_ipynb.py` to extract reusable functions/classes into `src/`.
- Keep exploratory code and plotting in `notebooks/` or `examples/`.
- To strip outputs from notebooks: install pre-commit and run `pre-commit run --all-files`.

## Development

- Run tests: `pytest -q --maxfail=1 --disable-warnings --cov=src`
- Lint: `ruff check .` | Format: `black .` | Type: `mypy src`

## Notes

- The repo previously contained only notebooks (e.g., `classifier_finetune_v7.ipynb`, `test_inference_v2.ipynb`). Those are preserved under `notebooks/` for education and experiments, not as core entry points.
- `--min_len` – minimum text length after cleaning
