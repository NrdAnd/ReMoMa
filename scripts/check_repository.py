#!/usr/bin/env python3
"""Check maintained source syntax, notebook cleanliness, configurations, and links."""
from __future__ import annotations

import ast
import json
from pathlib import Path
import re
import sys
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from remoma.config import load_config


def main() -> int:
    errors = []
    python_files = [p for folder in ("src", "scripts", "gnn/scripts", "archive")
                    for p in (ROOT / folder).rglob("*.py")]
    for path in python_files:
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as exc:
            errors.append(f"{path.relative_to(ROOT)}: {exc}")
    notebooks = list((ROOT / "notebooks").rglob("*.ipynb"))
    for path in notebooks:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            if cell.get("outputs") or cell.get("execution_count") is not None:
                errors.append(f"{path.relative_to(ROOT)} cell {index}: clear outputs and execution count")
            source = "".join(cell.get("source", []))
            # Shell/IPython cells require their kernel and are not Python modules.
            if any(line.lstrip().startswith(("!", "%")) for line in source.splitlines()):
                continue
            try:
                ast.parse(source, filename=f"{path.name}:cell{index}")
            except SyntaxError as exc:
                errors.append(str(exc))
    configs = [p for p in (ROOT / "configs").rglob("*.yaml") if "local" not in p.parts]
    for path in configs:
        try:
            cfg = load_config(path)
            if not Path(cfg["data"]["adj_matrix_path"]).is_file():
                errors.append(f"{path.relative_to(ROOT)}: missing reference graph")
        except (ValueError, KeyError) as exc:
            errors.append(f"{path.relative_to(ROOT)}: {exc}")
    markdown = [*ROOT.glob("*.md"), *(ROOT / "docs").rglob("*.md"),
                *(ROOT / "configs").rglob("*.md"), *(ROOT / "notebooks").rglob("*.md"),
                *(ROOT / "data").glob("*.md"), *(ROOT / "data/graphs").glob("*.md"),
                ROOT / "gnn/README.md", ROOT / "archive/README.md"]
    for path in markdown:
        text = path.read_text(encoding="utf-8")
        if text.count("```") % 2:
            errors.append(f"{path.relative_to(ROOT)}: unclosed fenced block")
        for target in re.findall(r'\]\(([^)]+)\)', text):
            target = target.strip("<>").split("#", 1)[0]
            if not target or "://" in target or target.startswith("mailto:"):
                continue
            if not (path.parent / unquote(target)).exists():
                errors.append(f"{path.relative_to(ROOT)}: broken local link {target}")
    for path in python_files + configs + markdown:
        if re.search(r"^(<<<<<<< |>>>>>>> |=======$)", path.read_text(), re.MULTILINE):
            errors.append(f"{path.relative_to(ROOT)}: unresolved merge marker")
    if errors:
        print("\n".join(errors))
        return 1
    print(f"Validated {len(python_files)} Python files, {len(notebooks)} notebooks, "
          f"{len(configs)} configurations, and {len(markdown)} Markdown documents.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
