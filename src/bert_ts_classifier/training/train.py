from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from ..data.datamodule import DataConfig, DataModule
from ..models.bert_classifier import BertClassifier, ModelConfig
from ..utils.io import load_yaml
from ..utils.logging import setup_logger
from ..utils.seed import set_seed


def _parse_overrides(overrides: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for ov in overrides:
        if "=" not in ov:
            continue
        key, val = ov.split("=", 1)
        # cast basic types
        if val.lower() in {"true", "false"}:
            cv: Any = val.lower() == "true"
        else:
            try:
                if "." in val:
                    cv = float(val)
                else:
                    cv = int(val)
            except ValueError:
                cv = val
        # nested dict update via dot path
        cur = out
        parts = key.split(".")
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = cv
    return out


def _apply_overrides(cfg: dict[str, Any], ov: dict[str, Any]) -> dict[str, Any]:
    for k, v in ov.items():
        if isinstance(v, dict) and k in cfg and isinstance(cfg[k], dict):
            cfg[k] = _apply_overrides(cfg[k], v)
        else:
            cfg[k] = v
    return cfg


def build_tokenizer(model_name: str):
    try:
        from transformers import AutoTokenizer  # noqa: PLC0415
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Transformers not installed. Install to use HF models.") from e

    tok = AutoTokenizer.from_pretrained(model_name)

    def tokenizer_fn(texts: list[str], max_len: int):
        enc = tok(
            texts,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}

    return tokenizer_fn


def train_one_epoch(model, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device):
    model.train()
    total = 0.0
    for batch in loader:
        optimizer.zero_grad()
        for k in ["input_ids", "attention_mask", "labels"]:
            batch[k] = batch[k].to(device)
        out = model(**{k: batch[k] for k in ["input_ids", "attention_mask", "labels"]})
        loss = out["loss"]
        loss.backward()
        optimizer.step()
        total += float(loss.item())
    return total / max(1, len(loader))


def evaluate(model, loader: DataLoader, device: torch.device):
    model.eval()
    all_logits, all_labels = [], []
    with torch.inference_mode():
        for batch in loader:
            for k in ["input_ids", "attention_mask", "labels"]:
                batch[k] = batch[k].to(device)
            out = model(**{k: batch[k] for k in ["input_ids", "attention_mask", "labels"]})
            all_logits.append(out["logits"].cpu())
            all_labels.append(batch["labels"].cpu())
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    preds = logits.argmax(-1)
    acc = (preds == labels).float().mean().item()
    return {"accuracy": acc}


def main():
    parser = argparse.ArgumentParser(description="Train classifier")
    parser.add_argument("--config", type=str, default="configs/train.yaml")
    parser.add_argument("overrides", nargs="*", help="key=value overrides using dot paths")
    args = parser.parse_args()

    base_cfg = load_yaml(args.config)
    ov = _parse_overrides(args.overrides)
    cfg = _apply_overrides(base_cfg, ov)

    log = setup_logger(__name__)
    log.info(f"Training with config: {cfg}")

    set_seed(cfg["data"].get("seed", 42))

    # Build components
    dcfg = DataConfig(**cfg["data"])  
    mcfg = ModelConfig(**cfg["model"])  

    tokenizer_fn = build_tokenizer(mcfg.model_name)

    dm = DataModule(dcfg, tokenizer_fn)
    dm.setup()

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    log.info(f"Using device: {device}")

    model = BertClassifier(mcfg).to(device)
    optimizer = AdamW(model.parameters(), lr=cfg["train"]["lr"]) 

    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()

    best_acc = 0.0
    for epoch in range(cfg["train"]["max_epochs"]):  
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        metrics = evaluate(model, val_loader, device)
        dt = time.time() - t0
        log.info(f"epoch={epoch+1} loss={train_loss:.4f} val_acc={metrics['accuracy']:.4f} dt={dt:.1f}s")
        best_acc = max(best_acc, metrics["accuracy"])  # noqa: PLW2901

    out_dir = Path(cfg["train"]["output_dir"]).resolve()  
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "model.pt")
    log.info(f"Saved weights to {out_dir / 'model.pt'} | best_acc={best_acc:.4f}")


if __name__ == "__main__":  # pragma: no cover
    main()
