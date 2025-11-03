# Changelog

## v0.1.0 — Notebook → Modules
- Introduced `src/bert_ts_classifier` package with data, models, training, evaluation, infer, utils.
- Added CLI entries runnable via `python -m bert_ts_classifier.training.train` and `...evaluation.eval`.
- Centralized configuration under `configs/` with YAML files.
- Added tests (data/model/metrics), CI workflows, and pre-commit hooks.
- Provided synthetic dataset generator to enable full local runs without private data.
- Added FastAPI service and export (ONNX/TorchScript) utilities.
- Preserved notebooks for exploration under `notebooks/` and added a converter script.
