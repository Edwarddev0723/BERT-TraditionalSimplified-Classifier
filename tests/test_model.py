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


def test_simple_classifier_attention_mask():
    cfg = ModelConfig(use_transformers=False, num_labels=2)
    model = BertClassifier(cfg)
    model.eval()

    # Input with padding (token 0)
    input_ids = torch.tensor([[1, 2, 3, 4, 0, 0], [5, 6, 0, 0, 0, 0]], dtype=torch.long)

    # Without attention mask
    out_no_mask = model(input_ids=input_ids)
    logits_no_mask = out_no_mask["logits"]

    # With attention mask
    attention_mask = torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 0, 0, 0, 0]], dtype=torch.long)
    out_with_mask = model(input_ids=input_ids, attention_mask=attention_mask)
    logits_with_mask = out_with_mask["logits"]

    # The bug is that attention_mask is ignored, so the logits will be identical.
    # This assertion will fail. After the fix, it should pass.
    assert not torch.allclose(logits_no_mask, logits_with_mask)
