from __future__ import annotations

import torch

from bert_ts_classifier.models.bert_classifier import BertClassifier, ModelConfig


def test_model_forward_shapes():
    cfg = ModelConfig(use_transformers=False, num_labels=2)
    model = BertClassifier(cfg)

    input_ids = torch.randint(0, 100, (4, 16))
    out = model(input_ids=input_ids, attention_mask=None, labels=torch.randint(0, 2, (4,)))

    assert out["logits"].shape == (4, 2)
    assert out["loss"] is not None
