"""Generate the molecule split and train-only energy statistics; do not copy data."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.torsion_data import prepare_data, save_protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT.parent / "datasets/torsion")
    parser.add_argument(
        "--out", type=Path, default=ROOT.parent / "outputs/torsion_protocol"
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    _, _, split, normalization = prepare_data(args.data, args.seed)
    save_protocol(args.out, split, normalization)
    print(
        f"Generated {split['counts']['train_molecules']}/{split['counts']['validation_molecules']} molecule split in {args.out}"
    )


if __name__ == "__main__":
    main()
