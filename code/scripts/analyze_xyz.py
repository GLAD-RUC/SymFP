"""Continuous target-conditioned PF, CSoM and Engel scores for one XYZ frame."""

import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch
from pymatgen.core import Element

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from recompute_pf import compute
from baseline.csom_modules.groups import hm_label, normalize_group_label
from baseline.csom_modules.core import LabeledPointCloud
from baseline.csom import CSoM
from baseline.engel import EngelGap


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("xyz", type=Path)
    p.add_argument("--target", required=True, help="Point group, e.g. C4 or 4.")
    p.add_argument("--frame", type=int, default=0)
    p.add_argument("--method", choices=["symfp", "csom", "engel"], default="symfp")
    p.add_argument(
        "--orientation",
        choices=["given", "fit"],
        default=None,
        help="Default: fit for Engel, given for CSoM/SymFP. Engel fit maximizes retained energy.",
    )
    p.add_argument(
        "--no-csom-start",
        action="store_true",
        help="Engel fit: omit the extra coordinate-fitted start; retain the Haar/BFGS search.",
    )
    p.add_argument("--neighbors", type=int, default=24)
    args = p.parse_args()
    if args.orientation is None:
        args.orientation = "fit" if args.method == "engel" else "given"
    if args.no_csom_start and (args.method != "engel" or args.orientation != "fit"):
        p.error("--no-csom-start requires --method engel with orientation fitting")
    if args.frame < 0 or args.neighbors < 1:
        p.error("frame must be nonnegative and neighbors positive")
    with args.xyz.open() as f:
        for _ in range(args.frame + 1):
            line = f.readline()
            if not line:
                p.error("frame exceeds the XYZ trajectory length")
            n = int(line)
            f.readline()
            rows = [f.readline().split() for _ in range(n)]
    if n < 2:
        p.error("at least two atoms are required")
    species = tuple(row[0] for row in rows)
    pos = np.array([[float(v) for v in row[1:4]] for row in rows])
    if pos.shape != (n, 3) or not np.isfinite(pos).all():
        p.error("invalid XYZ coordinates")
    group = normalize_group_label(args.target)
    torch.set_num_threads(1)
    if args.method == "symfp":
        target = hm_label(group)
        if target is None:
            p.error(
                "generic PF example supports the 32 crystallographic groups; use recompute_pf.py for E18"
            )
        if args.orientation != "given":
            p.error("SymFP does not require orientation fitting")
        z = np.array([Element(s).Z for s in species])
        result = compute(pos, z, target, k=args.neighbors)
    else:
        sample = LabeledPointCloud(args.xyz.stem, species, pos)
        if args.method == "csom":
            model = CSoM(
                candidate_groups=[group],
                center_mode="centroid",
                orientation_mode=(
                    "none" if args.orientation == "given" else "official_search"
                ),
            )
            score = model.evaluate_sample(sample).candidate_results[0]
            result = dict(
                CSoM=score.mean_deviation,
                implementation="official" if model.uses_official else "bundled",
            )
        else:
            model = EngelGap(
                candidate_groups=[group],
                center_mode="centroid",
                axis_mode="none" if args.orientation == "given" else "engel_search",
                use_csom_start=not args.no_csom_start,
            )
            score = model.evaluate_sample(sample).candidate_results[0]
            result = dict(
                Engel_score=score.engel_score,
                orientation_mode=score.orientation_mode,
                rotation_matrix=score.rotation_matrix.tolist(),
            )
            if args.orientation == "fit":
                result.update(
                    search_seed=model.config.orientation_seed,
                    haar_rotations=model.config.num_orientation_samples,
                    csom_start=model.config.use_csom_start,
                )
    print(
        json.dumps(
            dict(
                method=args.method,
                target=group,
                frame=args.frame,
                orientation=args.orientation,
                **result
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
