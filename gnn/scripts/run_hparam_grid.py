#!/usr/bin/env python3
"""Compatibility entry point; prefer the repository-root scripts directory."""
import os
import runpy
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[2]
for index, value in enumerate(sys.argv[1:], 1):
    if value.startswith("config/") and value.endswith(".yaml"):
        name = Path(value).name
        family = "recurrent" if name.startswith("recurrent") else "gnn"
        sys.argv[index] = str(root / "configs" / family / name)
    elif index > 1 and sys.argv[index - 1] in {"--checkpoint", "--checkpoint-dir", "--thresholds", "--base-config", "--config"}:
        sys.argv[index] = str(Path(value).resolve())
os.chdir(root)
if __name__ == "__main__":
    runpy.run_path(str(root / "scripts" / Path(__file__).name), run_name="__main__")
