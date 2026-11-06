"""FP32/FP64 runtime measurements for SymFP."""

import os

for k in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[k] = "1"
import sys
import time
import json
import hashlib
import types
import subprocess
import statistics
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from scripts import computational_efficiency as base

OUT = ROOT.parent / "outputs/computational_cost"
INPUT = ROOT.parent / "datasets/computational_cost"


def install(precision):
    dtype = getattr(torch, precision)
    # Initialize e3nn and every floating buffer in the requested dtype.
    torch.set_default_dtype(dtype)
    if precision == "float64":
        src = (ROOT / "models/para_free.py").read_text()
        src = src.replace("torch.float32", "torch.float64")
        snapshot = OUT / "para_free_fp64.py"
        snapshot.write_text(src)
        mod = types.ModuleType("models._timing_fp64")
        mod.__package__ = "models"
        sys.modules[mod.__name__] = mod
        exec(compile(src, str(snapshot), "exec"), mod.__dict__)
        base.ParaFreeModel = mod.ParaFreeModel
    original_make = base.make_net

    def make(groups, device):
        return original_make(groups, device).to(dtype=dtype)

    base.make_net = make

    def graph(p, z, system, device):
        if system == "ha":
            p = torch.tensor(p - p.mean(0), dtype=dtype)
        else:
            p = torch.tensor(p, dtype=dtype)
            if system != "dota":
                p -= p.mean(0)
        if system == "dota":
            ei = (~torch.eye(len(p), dtype=torch.bool)).nonzero().T
        else:
            ei = base.build_knn_edge_index(p, k=24, include_ties=system != "ha")
        d = base.Data(
            pos=p,
            x=torch.tensor(z, dtype=dtype)[:, None],
            edge_index=ei,
            num_nodes=len(p),
        )
        d.edge_attr = torch.linalg.norm(p[ei[0]] - p[ei[1]], dim=1, keepdim=True)
        return d.to(device)

    base.graph = graph

    # Same SVD operation as original, with requested precision instead of forced float32.
    def spectrum(net, data):
        features = net._resolve_graph_sh(net.encoder(data))
        keys, blocks, _ = base.merge_irrep_blocks(features, net.hidden_irreps)
        return [torch.linalg.svdvals(blocks[k].to(dtype=dtype)) for k in keys]

    base.spectrum = spectrum
    base.data_input = lambda system: (
        np.load(INPUT / f"{system}.npz")["pos"],
        np.load(INPUT / f"{system}.npz")["z"],
    )
    base.OUT = OUT
    return dtype


def preflight(dtype, device):
    from torch.utils._python_dispatch import TorchDispatchMode

    observed = set()

    class Check(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            out = func(*args, **(kwargs or {}))

            def walk(x):
                if isinstance(x, torch.Tensor) and x.is_floating_point():
                    observed.add(str(x.dtype))
                elif isinstance(x, (list, tuple)):
                    for v in x:
                        walk(v)
                elif isinstance(x, dict):
                    for v in x.values():
                        walk(v)

            walk(out)
            return out

    p, z = base.data_input("dota")
    net = base.make_net(base.GROUPS, device)
    with torch.inference_mode(), Check():
        data = base.graph(p, z, "dota", device)
        sv = base.spectrum(net, data)
        ls, pls = net._compute_soft_scores(sv)
    if dtype == torch.float64:
        assert observed == {str(dtype)}, observed
    assert data.pos.dtype == data.x.dtype == data.edge_attr.dtype == dtype
    assert all(v.dtype == dtype for v in sv) and pls.dtype == dtype
    assert torch.isfinite(pls).all()
    return {
        "floating_operation_output_dtypes": sorted(observed),
        "svd_dtypes": [str(v.dtype) for v in sv],
        "score_dtype": str(pls.dtype),
        "values": pls.cpu().tolist(),
    }


def scan(device):
    arr = dict(np.load(INPUT / "scan.npz"))
    windows = json.loads((INPUT / "windows.json").read_text())
    expected = json.loads((INPUT / "expected_scan.json").read_text())
    lookup = {(r["pdb"], r["region"]): r["pLS"] for r in expected}
    net = base.make_net(["C6"], device)
    stages = {}

    def call():
        values = []
        errors = []
        counts = []
        with torch.inference_mode():
            for pdb in ["8FCL", "8FCM", "8FCR", "8FCQ", "8FCO", "8FCT"]:
                for w in windows:
                    mask = (arr["resid"] >= w["start"]) & (arr["resid"] <= w["end"])
                    d = base.graph(arr[pdb][mask], arr["z"][mask], "p97", device)
                    sv = base.spectrum(net, d)
                    _, pls = net._compute_soft_scores(sv)
                    v = float(pls[0, 0].cpu())
                    values.append(v)
                    errors.append(abs(v - lookup[pdb, w["region"]]))
                    counts.append(int(mask.sum()))
        assert len(values) == 156 and np.isfinite(values).all()
        stages.update(
            readouts=156,
            max_archived_score_error=max(errors),
            n_min=min(counts),
            n_max=max(counts),
        )
        return values

    return call, stages, net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--precision", choices=["float32", "float64"], default="float64")
    ap.add_argument("--system", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--targets", type=int, default=1)
    a = ap.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision("highest")
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    OUT.mkdir(parents=True, exist_ok=True)
    dtype = install(a.precision)
    check = preflight(dtype, a.device)
    job = f"{a.precision}__{a.system}__{a.device}__{a.targets}"
    if a.system == "scan":
        call, stages, net = scan(a.device)
    else:
        call, stages, net = base.build_function(
            "SymFP",
            a.system,
            a.device,
            groups=base.GROUPS[: a.targets] if a.system == "dota" else None,
        )
    print("START", job, flush=True)
    result = base.measure(call, stages, a.device, job)
    result.update(
        precision=a.precision,
        system=a.system,
        device=a.device,
        targets=a.targets,
        dtype_validation=check,
    )
    if a.system != "scan":
        p, z = base.data_input(a.system)
        d = base.graph(p, z, a.system, "cpu")
        result["graph"] = {
            "nodes": len(p),
            "edges": d.edge_index.shape[1],
            "edge_hash": hashlib.sha256(d.edge_index.numpy().tobytes()).hexdigest(),
        }
    base.save(OUT / "jobs" / f"{job}.json", result)
    print("DONE", job, result["median_s"], flush=True)


if __name__ == "__main__":
    main()
