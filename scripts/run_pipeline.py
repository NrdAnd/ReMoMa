#!/usr/bin/env python3
"""Plan or run CSCO experiments from raw data through training-only NMI/TMFG and final metrics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from remoma.models import model_names
from remoma.pipeline.specification import load_spec, make_plan
from remoma.pipeline.runner import execute


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pipeline/cisco.yaml")
    parser.add_argument("--mode", choices=["by_file", "walk_forward"])
    parser.add_argument("--models", nargs="+", choices=model_names())
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--nmi-method", choices=["relative", "full"])
    parser.add_argument("--nmi-backend", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--raw-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--cache-dir")
    parser.add_argument("--stage", choices=["all", "graphs"], default="all")
    parser.add_argument("--run", action="store_true", help="Execute. Without this flag, print a read-only plan.")
    parser.add_argument("--resume", action="store_true", help="Validate the frozen experiment and skip completed stages.")
    parser.add_argument("--restart-incomplete", action="store_true", help="Archive interrupted model attempts and retrain them.")
    parser.add_argument("--quiet-progress", action="store_true", help="Hide NMI progress bars.")
    args = parser.parse_args()
    spec = load_spec(args.config)
    for key in ("mode", "models", "seeds", "raw_dir", "output_dir", "cache_dir"):
        if getattr(args, key) is not None:
            spec[key] = getattr(args, key)
    for key in ("method", "backend"):
        value = getattr(args, f"nmi_{key}")
        if value is not None:
            spec.setdefault("nmi", {})[key] = value
    if args.restart_incomplete and not args.resume:
        parser.error("--restart-incomplete requires --resume")
    plan = make_plan(spec)
    printable = {key: value for key, value in plan.items() if key != "configurations"}
    print(json.dumps(printable, indent=2))
    if not args.run:
        print("Plan only: no raw contents read, NMI computed, or training started. Add --run on your compute host.")
        return
    execute(spec, resume=args.resume, stage=args.stage, restart_incomplete=args.restart_incomplete,
            progress=not args.quiet_progress)


if __name__ == "__main__":
    main()
