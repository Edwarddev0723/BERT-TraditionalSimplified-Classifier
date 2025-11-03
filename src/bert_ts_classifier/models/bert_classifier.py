from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

try:
    # Optional: use existing focal loss if present
    from focal_loss import FocalLoss as ExternalFocalLoss
except Exception:  # pragma: no cover
    ExternalFocalLoss = None  # type: ignore


@dataclass
class ModelConfig:
    model_name: str = "ckiplab/bert-base-chinese"
    num_labels: int = 2
    dropout: float = 0.1
    use_transformers: bool = True
    use_focal_loss: bool = False


class SimpleClassifier(nn.Module):
    """A light-weight fallback classifier (for tests/offline).

    Embeds with bag-of-words mean (pretend) and a linear classifier.
    """

    def __init__(self, vocab_size: int = 30522, hidden_size: int = 128, num_labels: int = 2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None, labels: torch.Tensor | None = None):
        x = self.embed(input_ids)
        x = x.mean(dim=1)
        logits = self.classifier(x)
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
        return {"loss": loss, "logits": logits}


class BertClassifier(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.num_labels = cfg.num_labels
        self.classifier: nn.Module
        self.encoder: nn.Module | None
        self.loss_fn: nn.Module

        if cfg.use_transformers:
            try:
                from transformers import AutoConfig as _AutoConfig, AutoModel as _AutoModel  # noqa: PLC0415
            except Exception as e:  # pragma: no cover
                raise RuntimeError("Transformers not installed. Install to use HF models.") from e
            tcfg = _AutoConfig.from_pretrained(cfg.model_name, num_labels=cfg.num_labels)
            self.encoder = _AutoModel.from_pretrained(cfg.model_name, config=tcfg)
            hidden = tcfg.hidden_size
            self.dropout = nn.Dropout(cfg.dropout)
            self.classifier = nn.Linear(hidden, cfg.num_labels)
            self.uses_hf = True
        else:
            self.encoder = None
            self.classifier = SimpleClassifier(num_labels=cfg.num_labels)
            self.uses_hf = False

        # Loss
        if cfg.use_focal_loss and ExternalFocalLoss is not None:
            self.loss_fn = ExternalFocalLoss(alpha=0.5, gamma=2.0)
        else:
            self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None, labels: torch.Tensor | None = None):
        if self.uses_hf:
            assert self.encoder is not None
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            pooled = out.last_hidden_state[:, 0]
            logits = self.classifier(self.dropout(pooled))
        else:
            model_out = self.classifier(input_ids)
            logits = model_out["logits"]

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)
        return {"loss": loss, "logits": logits}
