#!/usr/bin/env python3
"""
Convert .ipynb notebooks into modular Python files.

Heuristics:
- Extract function/class definitions into candidate modules under src/.
- Keep plotting and shell commands in notebooks/examples.
- Write a report of extracted symbols.

Usage:
  python scripts/convert_ipynb.py path/to/notebook.ipynb --out src/bert_ts_classifier
"""
from __future__ import annotations

import argparse
import ast
from pathlib import Path
from typing import Dict, List, Tuple

import nbformat

SAFE_HEADERS = ("import ", "from ")


def is_def_cell(source: str) -> bool:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    return any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) for node in tree.body)


def split_cells(nb_path: Path) -> Tuple[List[str], List[str]]:
    nb = nbformat.read(nb_path, as_version=4)
    defs, others = [], []
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        src = cell.source
        if is_def_cell(src) or any(src.strip().startswith(h) for h in SAFE_HEADERS):
            defs.append(src)
        else:
            others.append(src)
    return defs, others


def write_module(defs: List[str], out_dir: Path, name: str = "from_notebook.py") -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / name
    header = "# Auto-generated from notebook. Review and refactor.\n\n"
    out_path.write_text(header + "\n\n".join(defs), encoding="utf-8")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("notebook")
    ap.add_argument("--out", default="src/bert_ts_classifier")
    args = ap.parse_args()

    nb_path = Path(args.notebook)
    out_dir = Path(args.out)

    defs, others = split_cells(nb_path)
    mod_path = write_module(defs, out_dir)

    report = {
        "notebook": nb_path.as_posix(),
        "module": mod_path.as_posix(),
        "defs_count": len(defs),
        "others_count": len(others),
    }
    print(report)


if __name__ == "__main__":  # pragma: no cover
    main()
