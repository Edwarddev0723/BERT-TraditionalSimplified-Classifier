from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from ..models.bert_classifier import BertClassifier, ModelConfig
from ..training.train import build_tokenizer
from ..utils.io import load_yaml


def main():
    parser = argparse.ArgumentParser(description="Batch inference")
    parser.add_argument("--config", type=str, default="configs/eval.yaml")
    parser.add_argument("--input", type=str, help="Input CSV with 'text' column")
    parser.add_argument("--output", type=str, default="preds.csv")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    mcfg = ModelConfig(**cfg["model"])  

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    model = BertClassifier(mcfg).to(device).eval()

    weights = cfg.get("eval", {}).get("weights", None)
    if weights:
        model.load_state_dict(torch.load(weights, map_location=device))

    tokenizer_fn = build_tokenizer(mcfg.model_name)

    texts: list[str] = []
    with open(args.input, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            texts.append(row["text"])

    preds: list[int] = []
    for i in range(0, len(texts), 64):
        batch = texts[i : i + 64]
        toks = tokenizer_fn(batch, cfg["data"]["max_len"])  
        for k in toks:
            toks[k] = toks[k].to(device)
        with torch.inference_mode():
            logits = model(**toks)["logits"]
            preds.extend(logits.argmax(-1).cpu().tolist())

    out_path = Path(args.output)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "pred"])
        for t, p in zip(texts, preds):  # noqa: B905
            writer.writerow([t, p])

    print(f"Saved predictions to {out_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
