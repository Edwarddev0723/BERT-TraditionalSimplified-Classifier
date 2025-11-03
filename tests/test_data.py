from __future__ import annotations

from pathlib import Path

from bert_ts_classifier.data.datamodule import DataConfig, DataModule
from bert_ts_classifier.training.train import build_tokenizer


def test_datamodule_collate(tmp_path: Path):
    # Prepare tiny data
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "train.jsonl").write_text('{"text": "軟體，影片", "label": 1}\n', encoding="utf-8")
    (data_dir / "val.jsonl").write_text('{"text": "軟件，視頻", "label": 0}\n', encoding="utf-8")

    cfg = DataConfig(train_path=str(data_dir / "train.jsonl"), val_path=str(data_dir / "val.jsonl"), batch_size=1, max_len=8, num_workers=0)
    tok = build_tokenizer("bert-base-chinese")
    dm = DataModule(cfg, tok)
    dm.setup()

    batch = next(iter(dm.train_dataloader()))
    assert batch["input_ids"].shape[0] == 1
    assert batch["labels"].shape == (1,)
