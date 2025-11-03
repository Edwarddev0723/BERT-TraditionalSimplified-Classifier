from __future__ import annotations

import csv
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch
from torch.utils.data import DataLoader, Dataset

from ..utils.seed import set_seed


class TextLabelDataset(Dataset):
    """Simple text classification dataset.

    Expects a JSONL or CSV with columns: text, label (int or str).
    """

    def __init__(self, items: list[tuple[str, int]]):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | int | str]:
        text, label = self.items[idx]
        return {"text": text, "label": label}


def _load_items(path: Path) -> list[tuple[str, int]]:
    items: list[tuple[str, int]] = []
    if path.suffix.lower() in {".jsonl", ".json"}:
        with open(path, encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                text = obj["text"]
                label = int(obj["label"]) if not isinstance(obj["label"], int) else obj["label"]
                items.append((text, label))
    elif path.suffix.lower() in {".csv"}:
        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                text = row["text"]
                label = int(row["label"]) if not isinstance(row["label"], int) else row["label"]
                items.append((text, label))
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")
    return items


@dataclass
class DataConfig:
    train_path: str = "data/train.jsonl"
    val_path: str = "data/val.jsonl"
    batch_size: int = 16
    num_workers: int = 2
    max_len: int = 256
    seed: int = 42


class DataModule:
    """Minimal datamodule to load text/label pairs and build tokenized batches."""

    def __init__(
        self,
        cfg: DataConfig,
        tokenizer_fn: Callable[[list[str], int], dict[str, torch.Tensor]],
    ) -> None:
        self.cfg = cfg
        self.tokenizer_fn = tokenizer_fn
        set_seed(cfg.seed)

        self.train_dataset: TextLabelDataset | None = None
        self.val_dataset: TextLabelDataset | None = None

    def setup(self) -> None:
        train_items = _load_items(Path(self.cfg.train_path))
        val_items = _load_items(Path(self.cfg.val_path))
        self.train_dataset = TextLabelDataset(train_items)
        self.val_dataset = TextLabelDataset(val_items)

    def collate_fn(self, batch: list[dict[str, torch.Tensor | int | str]]) -> dict[str, torch.Tensor]:
        texts: list[str] = [cast(str, ex["text"]) for ex in batch]
        labels = torch.tensor([int(ex["label"]) for ex in batch], dtype=torch.long)
        toks = self.tokenizer_fn(texts, self.cfg.max_len)
        toks["labels"] = labels
        return toks

    def train_dataloader(self) -> DataLoader:
        assert self.train_dataset is not None
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        assert self.val_dataset is not None
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            collate_fn=self.collate_fn,
        )
