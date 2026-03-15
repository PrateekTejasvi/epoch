from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import INPUT_CSV, PROCESSED_DIR
from src.epoch.preprocessing import run_preprocessing


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run preprocessing for the student digital harm dataset.")
    parser.add_argument("--input-csv", type=str, default=str(INPUT_CSV))
    parser.add_argument("--out-dir", type=str, default=str(PROCESSED_DIR))
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    outputs = run_preprocessing(args.input_csv, args.out_dir)
    print("Preprocessing completed.")
    print(f"analysis_base: {Path(outputs['analysis_base']).resolve()}")
    print(f"model_matrix: {Path(outputs['model_matrix']).resolve()}")


if __name__ == "__main__":
    main()
