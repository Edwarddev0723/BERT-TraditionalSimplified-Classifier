from __future__ import annotations

from pathlib import Path

import nbformat


def clear_outputs(nb_path: Path) -> bool:
    if not nb_path.exists():
        print(f"Skip missing: {nb_path}")
        return False
    nb = nbformat.read(nb_path, as_version=4)
    modified = False
    for cell in nb.cells:
        if cell.get("outputs"):
            cell["outputs"] = []
            modified = True
        if "execution_count" in cell and cell["execution_count"] is not None:
            cell["execution_count"] = None
            modified = True
    if modified:
        nbformat.write(nb, nb_path)
        print(f"Cleared outputs: {nb_path}")
    else:
        print(f"No changes: {nb_path}")
    return modified


def main() -> None:
    changed = []
    for p in [
        Path("classifier_finetune_v7.ipynb"),
        Path("test_inference_v2.ipynb"),
    ]:
        if clear_outputs(p):
            changed.append(str(p))
    print({"changed": changed})


if __name__ == "__main__":
    main()
