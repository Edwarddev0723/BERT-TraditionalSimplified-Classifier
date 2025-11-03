from __future__ import annotations

from pathlib import Path

from bert_ts_classifier.utils.io import load_yaml, save_yaml
from bert_ts_classifier.utils.seed import set_seed


def test_yaml_roundtrip(tmp_path: Path):
    obj = {"a": 1, "b": {"c": True}}
    p = tmp_path / "x.yaml"
    save_yaml(obj, p)
    out = load_yaml(p)
    assert out == obj


def test_set_seed_runs():
    # Should not raise
    set_seed(123, deterministic=True)
