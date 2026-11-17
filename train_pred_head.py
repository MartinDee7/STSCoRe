#!/usr/bin/env python3
"""Numerical tuning through both generations and the shared fusion head."""
import sys
from stscore.train import main

if __name__ == "__main__":
    if any(arg == "--stage" or arg.startswith("--stage=") for arg in sys.argv[1:]):
        raise SystemExit("Use train_SCoRe.py to select another stage.")
    main(sys.argv[1:] + ["--stage", "head"])
