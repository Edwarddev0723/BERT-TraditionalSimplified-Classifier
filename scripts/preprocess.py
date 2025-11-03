#!/usr/bin/env python3
"""
Generate a tiny synthetic dataset to run end-to-end.
Writes data/train.jsonl and data/val.jsonl with text,label.
"""
from __future__ import annotations

from pathlib import Path
import json
import random

WORDS_CN_MAIN = ["軟件", "視頻", "程序", "計算機", "開發", "下載", "界面"]
WORDS_TW = ["軟體", "影片", "程式", "電腦", "開發", "下載", "介面"]


def make_sentence(words):
    import random

    L = random.randint(5, 15)
    return "，".join(random.choices(words, k=L))


def gen_split(n: int, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for i in range(n):
            label = random.randint(0, 1)
            words = WORDS_CN_MAIN if label == 0 else WORDS_TW
            text = make_sentence(words)
            f.write(json.dumps({"text": text, "label": label}, ensure_ascii=False) + "\n")


def main():
    random.seed(42)
    gen_split(200, Path("data/train.jsonl"))
    gen_split(80, Path("data/val.jsonl"))
    print("Wrote data/train.jsonl and data/val.jsonl")


if __name__ == "__main__":  # pragma: no cover
    main()
