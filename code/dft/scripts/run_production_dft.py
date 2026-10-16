#!/usr/bin/env python3
"""Manually invoked production DFT campaign entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPOSITORY), str(REPOSITORY / "src")]

from mol_infer_gnn_experiments.dft import run_engine  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--engine", choices=("psi4", "pyscf-geometric"), required=True)
    arguments = parser.parse_args()
    run_engine(arguments.manifest, arguments.output_root.resolve(), arguments.engine)


if __name__ == "__main__":
    main()
