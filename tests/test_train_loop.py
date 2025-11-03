from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset

from bert_ts_classifier.models.bert_classifier import BertClassifier, ModelConfig
from bert_ts_classifier.training.train import evaluate, train_one_epoch
from bert_ts_classifier.utils.logging import setup_logger


class TinyDS(Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, idx):
        return {
            "input_ids": torch.randint(0, 50, (8,)),
            "attention_mask": torch.ones(8, dtype=torch.long),
            "labels": torch.randint(0, 2, (1,)).squeeze(0),
        }


def collate(batch):
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch], 0),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch], 0),
        "labels": torch.stack([b["labels"] for b in batch], 0),
    }


def test_train_and_eval_cpu():
    setup_logger(__name__)
    device = torch.device("cpu")
    cfg = ModelConfig(use_transformers=False)
    model = BertClassifier(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    train_loader = DataLoader(TinyDS(), batch_size=2, collate_fn=collate)
    val_loader = DataLoader(TinyDS(), batch_size=2, collate_fn=collate)

    loss = train_one_epoch(model, train_loader, opt, device)
    metrics = evaluate(model, val_loader, device)

    assert loss >= 0
    assert 0.0 <= metrics["accuracy"] <= 1.0
