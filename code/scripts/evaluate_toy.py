"""Evaluate point-group identification and rank-template matching at a fixed tolerance."""

import argparse
import json
import sys
from pathlib import Path
import torch
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from utils.toy_data_utils import ToySymmetryDataModule
from utils.toy_model_utils import build_or_load_toy_encoder
from utils.rank_evaluation import (
    build_natural_parity_template,
    collect_sigmas,
    evaluate_at_tol,
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model",
        choices=["para_free", "hegnn", "tfn", "mace", "equiformer"],
        default="para_free",
    )
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--degree", type=int, choices=[6, 9], default=6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split", choices=["train", "val", "test"], default="test")
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument(
        "--limit",
        type=int,
        help="Optional smoke-test limit; omit for the complete split.",
    )
    p.add_argument("--output", type=Path, default=ROOT / "outputs/toy_evaluation.json")
    args = p.parse_args()
    if args.limit is not None and args.limit < 1:
        p.error("--limit must be positive")
    pl.seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision("highest")
    channels = 16 if args.degree == 6 else 24
    tolerance = 3e-5 if args.degree == 6 else 1e-6
    encoder, metadata = build_or_load_toy_encoder(
        model_type=args.model,
        ckpt_path=str(args.checkpoint) if args.checkpoint else None,
        max_ell=args.degree,
        irreps_channels=channels,
        device=args.device,
    )
    if metadata["max_ell"] != args.degree:
        p.error("Checkpoint degree does not match --degree")
    dm = ToySymmetryDataModule(
        pkl_path=str(ROOT / "datasets/toy/toy_symmetry.pkl"),
        processed_root=str(ROOT / "outputs/processed"),
        batch_size=args.batch_size,
        num_workers=0,
        model_type=args.model,
        test_noise_levels=[0.0],
    )
    dm.setup("test" if args.split == "test" else "fit")
    dataset = (
        dm.test_datasets[0]
        if args.split == "test"
        else getattr(dm, args.split + "_dataset")
    )
    if args.limit:
        dataset = dataset[: args.limit]
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    keys, template = build_natural_parity_template(args.degree)
    cache = collect_sigmas(
        encoder.eval(),
        dm,
        {args.split: loader},
        torch.device(args.device),
        keys,
        encoder.hidden_irreps,
    )
    scores = evaluate_at_tol(cache, template, tolerance, relative=True)[args.split]
    result = dict(
        model=args.model,
        degree=args.degree,
        seed=args.seed,
        split=args.split,
        relative_rank_tolerance=tolerance,
        checkpoint=args.checkpoint.name if args.checkpoint else None,
        point_group_accuracy=scores["accuracy"],
        rank_profile_accuracy=scores["exact_match_rate"],
        count=scores["count"],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
