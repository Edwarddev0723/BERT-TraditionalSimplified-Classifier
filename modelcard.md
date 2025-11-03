---
language:
    - en
    - zh
tags:
- text-classification
- chinese
- traditional-chinese
- bert
- pytorch
license: apache-2.0
datasets:
- custom
metrics:
- accuracy
- f1
library_name: transformers
pipeline_tag: text-classification
model-index:
- name: bert-traditional-chinese-classifier
  results:
  - task:
      type: text-classification
      name: Traditional Chinese Classification
    metrics:
    - type: accuracy
      value: 0.8771
      name: Accuracy
    - type: f1
      value: 0.8771
      name: F1 Score
base_model:
- ckiplab/bert-base-chinese
---

# BERT Traditional Chinese Classifier (Mainland vs. Taiwan)

A BERT-based classifier to distinguish Mainland Traditional vs. Taiwan Traditional Chinese usage.

## Model overview

- Base model: ckiplab/bert-base-chinese
- Task: Traditional Chinese text classification (Mainland vs Taiwan)
- Reported accuracy: 87.71% (validation)
- Training samples: ~156,824

## Highlights

- Handles long texts via sliding window (max length 384 tokens)
- Optional Focal Loss for class imbalance
- Optional multi-sample dropout and progressive unfreezing
- Layer-wise learning rate decay and AdamW optimizer

### Optimization and generalization

- Layer-wise LR decay (≈0.95), higher LR for the classification head
- AdamW (weight_decay=0.01, betas≈(0.9, 0.98))
- Optional OneCycleLR for fast warmup and stable convergence
- Early stopping and gradient accumulation

## Usage

```python
import torch, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --- Basic config ---
REPO_ID = "renhehuang/bert-traditional-chinese-classifier"
LABELS = {0: "Mainland Traditional", 1: "Taiwan Traditional"}
MAX_LEN, STRIDE = 384, 128

# --- Device ---
device = (
    "mps" if torch.backends.mps.is_available()
    else ("cuda" if torch.cuda.is_available() else "cpu")
)

# --- Load model & tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(REPO_ID, cache_dir=".cache")
model = AutoModelForSequenceClassification.from_pretrained(REPO_ID, cache_dir=".cache")
model.to(device).eval()

# --- Long-text chunking ---
def chunk_encode(text, max_len=MAX_LEN, stride=STRIDE):
    ids = tokenizer(text, add_special_tokens=False, return_attention_mask=False)["input_ids"]
    if len(ids) <= max_len - 2:
        enc = tokenizer(text, truncation=True, max_length=max_len,
                        return_attention_mask=True, return_tensors="pt")
        return [enc]
    enc = tokenizer(text, truncation=True, max_length=max_len, stride=stride,
                    return_overflowing_tokens=True, return_attention_mask=True,
                    return_tensors="pt")
    return [{"input_ids": enc["input_ids"][i:i+1],
             "attention_mask": enc["attention_mask"][i:i+1]}
            for i in range(len(enc["input_ids"]))]

# --- Single-text inference ---
@torch.inference_mode()
def predict(text: str):
    chunks = chunk_encode(text)
    probs_all = []
    for ch in chunks:
        logits = model(
            input_ids=ch["input_ids"].to(device),
            attention_mask=ch["attention_mask"].to(device)
        ).logits
        probs_all.append(F.softmax(logits, dim=-1).cpu())
    avg = torch.cat(probs_all, 0).mean(0)
    label_id = int(avg.argmax())
    return {
        "text_preview": (text[:100] + "...") if len(text) > 100 else text,
        "predicted_id": label_id,
        "predicted_name": LABELS[label_id],
        "confidence": float(avg[label_id]),
        "probabilities": {LABELS[0]: float(avg[0]), LABELS[1]: float(avg[1])},
        "num_chunks": len(chunks),
        "device": device,
    }

# --- Quick test ---
if __name__ == "__main__":
    tests = [
        "這個軟件的界面設計得很好。",
        "這個軟體的介面設計得很好。",
        "我需要下載這個程序到計算機上。",
        "我需要下載這個程式到電腦上。",
    ]
    for t in tests:
        r = predict(t)
        print(f"{r['predicted_name']} | conf={r['confidence']:.2%} | {r['text_preview']}")

```

## Long-text and robust inference (optional MC Dropout voting)

- Long texts are chunked with a sliding window (MAX_LEN=384, STRIDE=128) and averaged.
- For uncertainty estimation, run multiple stochastic passes (with dropout) and vote on labels; confidence is the mean probability of the voted class.

```python
from collections import Counter

@torch.inference_mode()
def predict_runs(text: str, n_runs: int = 3, enable_dropout: bool = True):
    # Pre-chunk
    chunks = chunk_encode(text)

    prev_training = model.training
    run_prob_list = []
    try:
    model.train() if enable_dropout else model.eval()  # enable MC Dropout
        for _ in range(n_runs):
            probs_all = []
            for ch in chunks:
                logits = model(
                    input_ids=ch["input_ids"].to(device),
                    attention_mask=ch["attention_mask"].to(device)
                ).logits
                probs_all.append(F.softmax(logits, dim=-1).cpu())
            run_prob_list.append(torch.cat(probs_all, 0).mean(0))
    finally:
        model.train() if prev_training else model.eval()

    probs_stack = torch.stack(run_prob_list, 0)
    per_run_ids = probs_stack.argmax(-1).tolist()
    vote_counts = Counter(per_run_ids)
    mean_probs = probs_stack.mean(0)

    # Majority vote + mean probability as a tie-breaker
    voted_id = max(vote_counts.items(), key=lambda kv: (kv[1], mean_probs[kv[0]].item()))[0]
    return LABELS[voted_id], float(mean_probs[voted_id]), dict(vote_counts)
```

## Labels

- 0 → Mainland Traditional
- 1 → Taiwan Traditional

## Data and training (summary)

- Aggregated Traditional Chinese corpora from multiple sources and lengths, balanced and quality-controlled.
- Tokenization: BERT WordPiece; long-text chunking with 384/128 sliding window.
- Loss: Focal Loss (gamma=2.0); optional light label smoothing (~0.05).
- Optimization: AdamW + layer-wise LR decay (~0.95); optional OneCycleLR.
- Regularization: Multi-sample dropout, progressive unfreezing, early stopping, gradient accumulation.

## Training configuration

- Batch Size: 16
- Learning Rate: 2e-05 (base), 4e-05 (head)
- Epochs: 4
- Max Length: 384
- Loss Function: Focal Loss (gamma=2.0)

## Evaluation metrics

- Overall: Accuracy / F1 ≈ 0.8771 (validation)
- Length-wise: Stable on very long texts after chunk averaging
- Typical confusions: mixed regional vocabulary and domain-specific jargon

> For full learning curves and diagnostic plots, see repository outputs.

## Intended use and limitations

Intended: origin-style identification, data cleaning, annotation assistance, pre-normalization, and hybrid use with rules/other models.

Limitations:
- Mixed regional usage, translations, heavy emoji/code/foreign text reduce confidence.
- Domain-specific jargon may bias results toward a region.
- Very short or heavily colloquial snippets are harder.

## Fairness, safety, and risk

- The model reflects the training data distribution; biases may exist for topics/domains.
- Do not use as a single source of truth; combine with human review or model ensembles.
- Follow local laws and platform policies.

## Deployment tips

- For critical paths, consider multiple runs (3–10) and a confidence threshold (e.g., ≥ 0.85).
- Route low-confidence samples to human review.
- Monitor domain shift and periodically fine-tune with new feedback data.

## Citation

If you use this model, please cite:

```
@misc{bert-traditional-chinese-classifier,
  author = {renhehuang},
  title = {BERT Traditional Chinese Classifier},
  year = {2025},
  publisher = {Hugging Face},
  howpublished = {\url{https://huggingface.co/renhehuang/bert-traditional-chinese-classifier}}
}
```

## License

Apache-2.0

## Contact

Please open an issue on the Hugging Face model page or GitHub repository.