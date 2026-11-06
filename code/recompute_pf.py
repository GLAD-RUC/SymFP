"""Recompute a published biological selection using the supplied coordinates."""

from pathlib import Path
import argparse
import json
import numpy as np
import torch
from scipy.spatial import cKDTree
from torch_geometric.data import Data
from models.para_free import ParaFreeModel
from models.symfp_wrapper import SymFPWrapper


def compute(pos, z, group, k=24, ha=False, degree=None, torch_graph=False):
    p = torch.as_tensor(pos, dtype=torch.float64 if ha else torch.float32)
    p = (p - p.mean(0)).float()
    if ha or torch_graph:
        pieces = []
        for start in range(0, len(p), 1024):
            distance = torch.cdist(p[start : start + 1024], p)
            distance[
                torch.arange(len(distance)), torch.arange(start, start + len(distance))
            ] = float("inf")
            if torch_graph:
                cutoff = torch.topk(
                    distance, min(k, len(p) - 1), dim=1, largest=False
                ).values[:, -1:]
                rows, cols = torch.where(distance <= cutoff + 1e-7 + 1e-6 * cutoff)
                pieces.append(torch.stack([rows + start, cols]))
            else:
                cols = torch.topk(
                    distance, min(k, len(p) - 1), dim=1, largest=False
                ).indices
                rows = torch.arange(start, start + len(distance))[:, None].expand_as(
                    cols
                )
                pieces.append(torch.stack([rows.flatten(), cols.flatten()]))
        edges = torch.cat(pieces, dim=1)
    else:
        tree = cKDTree(p.numpy())
        dist, _ = tree.query(p.numpy(), k=min(k + 1, len(p)), workers=1)
        last = dist[:, -1]
        neighbours = tree.query_ball_point(
            p.numpy(), last + 1e-7 + 1e-6 * last, workers=1
        )
        rows = []
        cols = []
        for i, js in enumerate(neighbours):
            js = sorted(j for j in js if j != i)
            rows.extend([i] * len(js))
            cols.extend(js)
        edges = torch.tensor([rows, cols], dtype=torch.long)
    net = SymFPWrapper(
        ParaFreeModel(irreps_channels=16, branch_mode="hybrid"),
        point_groups=["6" if group == "I" else group],
        ls_eps=1e-10,
    ).eval()
    if group == "I":
        for target in [net.observable_rank_matrix, net.theoretical_rank_matrix]:
            target.copy_(torch.tensor([[1, 0, 0, 0, 0, 0, 1]]))
    d = Data(
        pos=p,
        x=torch.as_tensor(z, dtype=torch.float32)[:, None],
        edge_index=edges,
        num_nodes=len(p),
    )
    d.edge_attr = (p[edges[0]] - p[edges[1]]).norm(dim=1, keepdim=True)
    with torch.inference_mode():
        r = net(d)
    ls = float(r["ls_matrix"][0, 0])
    pls = float(r["pls_matrix"][0, 0])
    if degree is not None:
        s = r["block_singular_values"][degree][0]
        m = int(net.observable_rank_matrix[0, degree])
        ls = float(s[m:].square().sum() / (s.square().sum() + 1e-10))
        pls = float(-np.log10(ls + 1e-10))
    return dict(atoms=len(p), edges=edges.shape[1], LS=ls, pLS=pls)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("system", choices=["p97", "ha", "gluk2", "e18"])
    ap.add_argument("--structure")
    ap.add_argument("--region", default="D1")
    ap.add_argument("--layer")
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = ap.parse_args()
    root = args.root
    torch.set_num_threads(1)
    if args.system == "p97":
        a = np.load(root / "datasets/p97/matched_coordinates.npz")
        region = next(
            r
            for r in json.loads((root / "datasets/p97/regions.json").read_text())
            if r["region"] == args.region
        )
        mask = (a["resid"] >= region["start"]) & (a["resid"] <= region["end"])
        result = compute(
            a[args.structure or "8FCL"][mask], a["z"][mask], "6", torch_graph=True
        )
    elif args.system == "ha":
        a = np.load(root / f"datasets/ha/{args.structure or '6Y5G'}.npz")
        mask = a[args.region if args.region in ["whole", "HA1", "HA2"] else "whole"]
        result = compute(a["pos"][mask], a["z"][mask], "3", ha=True)
    elif args.system == "gluk2":
        a = np.load(root / "datasets/gluk2/common_coordinates.npz")
        if args.layer:
            region = next(
                r
                for r in json.loads((root / "datasets/gluk2/layers.json").read_text())[
                    "windows"
                ]
                if r["id"] == args.layer
            )
            assert region["valid"]
            mask = region["atom_indices"]
        else:
            domains = json.loads((root / "datasets/gluk2/domains.json").read_text())
            ranges = domains.get(args.region, domains["ATD"])
            mask = np.logical_or.reduce(
                [(a["resid"] >= lo) & (a["resid"] <= hi) for lo, hi in ranges]
            )
        result = compute(a["pos"][mask], a["z"][mask], "4")
    else:
        a = np.load(root / f"datasets/e18/{args.structure or '6HBH'}.npz")
        result = compute(a["pos"], a["z"], "I", degree=6)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
