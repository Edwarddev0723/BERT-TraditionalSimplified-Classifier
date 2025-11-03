from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, auc, confusion_matrix, f1_score, precision_recall_curve

from ..data.datamodule import DataConfig, DataModule
from ..models.bert_classifier import BertClassifier, ModelConfig
from ..training.train import build_tokenizer
from ..utils.io import load_yaml
from ..utils.logging import setup_logger


def _save_confusion_matrix(cm: np.ndarray, labels: list[str], path: Path) -> None:
    import matplotlib.pyplot as plt  # noqa: PLC0415
    from matplotlib import cm as mpl_cm  # noqa: PLC0415

    fig, ax = plt.subplots(figsize=(4, 4))
    cmap = getattr(mpl_cm, "Blues")
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]), xticklabels=labels, yticklabels=labels, ylabel="True", xlabel="Pred")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    fmt = "d"
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt), ha="center", va="center", color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=150)
    plt.close(fig)


def main():
    BINARY_CLASS_COUNT = 2
    parser = argparse.ArgumentParser(description="Evaluate classifier")
    parser.add_argument("--config", type=str, default="configs/eval.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    log = setup_logger(__name__)

    dcfg = DataConfig(**cfg["data"])  
    mcfg = ModelConfig(**cfg["model"])  

    tokenizer_fn = build_tokenizer(mcfg.model_name)

    dm = DataModule(dcfg, tokenizer_fn)
    dm.setup()

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    model = BertClassifier(mcfg).to(device)

    # Load weights if provided
    weights = cfg.get("eval", {}).get("weights", None)
    if weights:
        model.load_state_dict(torch.load(weights, map_location=device))
        log.info(f"Loaded weights from {weights}")

    model.eval()
    all_logits, all_labels = [], []
    with torch.inference_mode():
        for batch in dm.val_dataloader():
            for k in ["input_ids", "attention_mask", "labels"]:
                batch[k] = batch[k].to(device)
            out = model(**{k: batch[k] for k in ["input_ids", "attention_mask", "labels"]})
            all_logits.append(out["logits"].cpu())
            all_labels.append(batch["labels"].cpu())

    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    preds = logits.argmax(-1)

    acc = float(accuracy_score(labels, preds))
    f1 = float(f1_score(labels, preds, average="macro"))
    # AUPRC for positive class if binary
    if logits.shape[1] == BINARY_CLASS_COUNT:
        prob1 = torch.softmax(torch.from_numpy(logits), -1)[:, 1].numpy()
        prec, rec, _ = precision_recall_curve(labels, prob1)
        auprc = float(auc(rec, prec))
    else:
        auprc = float("nan")

    cm = confusion_matrix(labels, preds)

    out_dir = Path(cfg.get("eval", {}).get("output_dir", "runs/eval")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = {"accuracy": acc, "f1_macro": f1, "auprc": auprc}
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    _save_confusion_matrix(cm, labels=["Mainland Trad", "Taiwan Trad"], path=out_dir / "confusion_matrix.png")

    log.info(f"Saved metrics to {out_dir}")


if __name__ == "__main__":  # pragma: no cover
    main()
