"""Verify checkpoints, extract pLS, or reproduce torsion pretraining tables.

Examples:
  python code/scripts/torsion_pretraining.py verify
  python code/scripts/torsion_pretraining.py analyze --scores outputs/torsion_scores --out outputs/torsion_reproduced
  python code/scripts/torsion_pretraining.py extract --model all --variant all --device cuda:0 --out outputs/torsion_scores
"""

from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(name, "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.torsion_pretraining_analysis import (
    MODELS,
    GROUPS,
    HM,
    load_manifest,
    contained_path,
    sha,
)


def trajectory(path):
    import numpy as np

    lines = Path(path).read_text().splitlines()
    cursor = 0
    frames = []
    while cursor < len(lines) and lines[cursor].strip():
        n = int(lines[cursor])
        rows = [line.split() for line in lines[cursor + 2 : cursor + 2 + n]]
        if len(rows) != n:
            raise ValueError(f"Truncated XYZ: {path}")
        frames.append(
            ([r[0] for r in rows], np.array([[float(x) for x in r[1:4]] for r in rows]))
        )
        cursor += n + 2
    return frames


def extract(args):
    import numpy as np
    import pandas as pd
    import torch
    from torch_geometric.data import Data
    from models.backbone_factory import build_backbone
    from models.symfp_wrapper import SymFPWrapper, format_signature_key

    manifest = load_manifest(args.root)
    from utils.torsion_data import prepare_data

    data_dir = args.root / manifest["dataset"]
    frames, _, _, _ = prepare_data(data_dir, manifest["seed"])
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    numbers = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9, "S": 16, "Cl": 17, "Br": 35}
    geometry = {
        rid: trajectory(contained_path(data_dir, group.trajectory_path.iloc[0]))
        for rid, group in frames.groupby("record_id")
    }
    assert all(len(value) == 24 for value in geometry.values())
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        frames = frames.iloc[: args.limit]
    for entry in manifest["entries"]:
        if args.model != "all" and entry["model"] != args.model:
            continue
        if args.variant != "all" and entry["variant"] != args.variant:
            continue
        out = args.out / entry["variant"] / f'{entry["model"]}.csv'
        if out.exists():
            raise FileExistsError(f"Refusing to replace scores: {out}")
        checkpoint = contained_path(args.root, entry["checkpoint"])
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        assert (
            state["model_type"] == entry["model"]
            and state["variant"] == entry["variant"]
        )
        backbone = build_backbone(**state["backbone_config"], device=args.device).eval()
        backbone.load_state_dict(state["backbone_state_dict"], strict=True)
        wrapper = (
            SymFPWrapper(backbone, ls_eps=1e-10, point_groups=[HM[g] for g in GROUPS])
            .to(args.device)
            .eval()
        )
        ranks = wrapper.observable_rank_matrix.cpu().numpy().astype(int)
        rows = []
        with torch.inference_mode():
            for index, frame in enumerate(frames.itertuples(), 1):
                symbols, coords = geometry[frame.record_id][frame.frame_index]
                atomic = torch.tensor([numbers[s] for s in symbols], dtype=torch.long)
                pos = torch.tensor(coords, dtype=torch.float64)
                pos = (pos - pos.mean(0, keepdim=True)).float()
                grid = torch.arange(len(pos))
                i, j = torch.meshgrid(grid, grid, indexing="ij")
                mask = i != j
                edges = torch.stack([i[mask], j[mask]])
                graph = Data(
                    x=torch.nn.functional.one_hot(atomic, num_classes=101).float(),
                    pos=pos,
                    edge_index=edges,
                    atomic_numbers=atomic,
                    num_nodes=len(pos),
                    edge_attr=torch.linalg.norm(
                        pos[edges[1]] - pos[edges[0]], dim=-1, keepdim=True
                    ),
                )
                output = wrapper(graph.to(args.device))
                names = [format_signature_key(k) for k in output["signature_keys"]]
                energies = [
                    s[0].float().cpu().square() for s in output["block_singular_values"]
                ]
                assert all(torch.isfinite(v).all() for v in energies)
                row = dict(
                    record_id=int(frame.record_id), frame_index=int(frame.frame_index)
                )
                for group, target in enumerate(GROUPS):
                    ratios = [
                        float(e[ranks[group, k] :].sum()) / (float(e.sum()) + 1e-10)
                        for k, e in enumerate(energies)
                        if names[k] in ("0e", "2e", "4e", "6e")
                    ]
                    row[f"pls_even_{target}"] = -np.log10(np.mean(ratios) + 1e-10)
                rows.append(row)
                if index % 240 == 0:
                    print(
                        entry["variant"],
                        entry["model"],
                        index,
                        "/",
                        len(frames),
                        flush=True,
                    )
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out, index=False)
        out.with_suffix(".json").write_text(
            json.dumps(
                dict(
                    model=entry["model"],
                    variant=entry["variant"],
                    epoch=entry["epoch"],
                    n_frames=len(rows),
                    complete=len(rows) == manifest["n_frames"],
                    checkpoint_sha256=sha(checkpoint),
                    device=args.device,
                    precision="FP32, TF32 disabled",
                ),
                indent=2,
            )
            + "\n"
        )
        print("Saved", out, flush=True)
        del wrapper, backbone
        if str(args.device).startswith("cuda"):
            torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("stage", choices=["verify", "extract", "analyze"])
    p.add_argument(
        "--root",
        type=Path,
        default=ROOT.parent,
        help="Repository root after extracting the archives",
    )
    p.add_argument("--out", type=Path)
    p.add_argument(
        "--scores",
        type=Path,
        help="Required for analyze: directory of extracted variant/model.csv scores",
    )
    p.add_argument("--model", choices=[*MODELS, "all"], default="all")
    p.add_argument(
        "--variant", choices=["random", "energy", "unimol", "all"], default="all"
    )
    p.add_argument("--device", default="cpu")
    p.add_argument(
        "--limit",
        type=int,
        help="Extraction smoke only; partial results cannot be analyzed",
    )
    args = p.parse_args()
    if args.stage == "analyze" and args.scores is None:
        p.error("--scores is required for analyze")
    args.root = args.root.resolve()
    if args.stage == "verify":
        manifest = load_manifest(args.root)
        print(
            f'Verified {len(manifest["files"])} files, {len(manifest["entries"])} checkpoints, {manifest["n_frames"]} conformations.'
        )
    else:
        if args.out is None:
            p.error("--out is required for extract/analyze")
        args.out = args.out.resolve()
        if args.stage == "extract":
            extract(args)
        else:
            from utils.torsion_pretraining_analysis import analyze

            analyze(args.root, args.out, args.scores)
            print("Analysis saved:", args.out)


if __name__ == "__main__":
    main()
