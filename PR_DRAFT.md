Title: Refactor: Notebook-only repo → standardized, installable, testable package

Summary

This PR converts the repository from notebook-centric to a standard Python package with CLI, configs, tests, CI, and serving/export utilities.

Before

- Notebooks only for training/eval/inference
- Minimal scripts; no packaging, tests, or CI
- No standardized configs or CLI

After

- Python package under src/bert_ts_classifier with:
  - Data module, model (BERT wrapper), training/eval CLIs, inference (batch + FastAPI), export (ONNX/TorchScript), and utilities (seed/logging/io)
  - Configs in configs/ (data/model/train/eval)
  - Scripts for preprocessing, notebook conversion, and notebook output clearing
  - Tests (pytest) achieving ~79% coverage
  - Docs: README, MODEL_CARD, CHANGELOG; pre-commit hooks; CI workflows

Repository tree (key parts)

- pyproject.toml
- .pre-commit-config.yaml
- .github/workflows/
  - quality.yml (ruff + mypy + pytest)
  - tests.yml (pytest)
- configs/
  - data.yaml, model.yaml, train.yaml, eval.yaml
- src/bert_ts_classifier/
  - data/datamodule.py
  - models/bert_classifier.py
  - training/train.py
  - evaluation/eval.py
  - infer/predict.py, infer/serve_fastapi.py
  - export/to_onnx.py, export/to_torchscript.py
  - utils/io.py, utils/logging.py, utils/seed.py
- scripts/
  - convert_ipynb.py
  - preprocess.py
  - clear_outputs.py
- tests/
  - test_data.py, test_model.py, test_metrics.py, test_train_utils.py, test_utils.py, test_train_loop.py
- notebooks/
  - quickstart.ipynb, inference.ipynb (outputs cleared)

Quickstart

Training

python -m bert_ts_classifier.training.train --config configs/train.yaml 

with overrides (dot notation):

python -m bert_ts_classifier.training.train --config configs/train.yaml train.max_epochs=2 data.batch_size=8 

Evaluation

python -m bert_ts_classifier.evaluation.eval --config configs/eval.yaml 

Batch inference

python -m bert_ts_classifier.infer.predict --config configs/eval.yaml --input path/to/texts.csv --output preds.csv 

Serve (FastAPI)

uvicorn bert_ts_classifier.infer.serve_fastapi:app --reload 

Export

python -m bert_ts_classifier.export.to_onnx 
python -m bert_ts_classifier.export.to_torchscript 

Quality gates (local)

- Ruff: PASS
- Mypy: PASS
- Pytest: PASS (coverage ~79%)
- Pre-commit: PASS

Notes

- The FastAPI app stores model/tokenizer/device in app.state; no module-level globals.
- Notebooks’ outputs are cleared; a script is provided to re-clear as needed.
- Python requires >=3.10;<3.13; hooks configured for Python 3.12.

Limitations & next steps

- Exporters are minimal; add a consistency check comparing ONNX/TorchScript vs. PyTorch logits on sample inputs.
- Consider wiring optional focal loss from focal_loss.py via config (currently auto-detected).
- If needed, integrate Gradio_app.py as a thin client pointing to FastAPI.
