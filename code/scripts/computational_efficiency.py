"""Controlled runtime measurements using the bundled coordinate inputs."""

from __future__ import annotations

import os

for key in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[key] = "1"

import sys
import time
import json
import hashlib
import signal
import resource
import contextlib
import io
import argparse
import subprocess
import statistics

from pathlib import Path

import numpy as np

import torch

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT))

OUT = ROOT.parent / "outputs/computational_cost"

from models.para_free import ParaFreeModel

from models.symfp_wrapper import SymFPWrapper

from models.symfp_hard import merge_irrep_blocks, block_singular_values

from utils.protein_chain_utils import build_knn_edge_index

from torch_geometric.data import Data

from baseline.csom import CSoM

from baseline.csom_modules.core import LabeledPointCloud, PreparedPointCloud

from baseline.csom_modules.groups import canonical_label_pair

import baseline.engel as engel

from e3nn import o3

GROUPS = ["C2", "C4", "C6", "D2", "D4", "D6", "O"]

CAP = 7200


def save(path, obj):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(p)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


class TimedOut(BaseException):
    pass


def alarm(*_):
    raise TimedOut()


def data_input(system):
    a = np.load(OUT / f"inputs/{system}.npz")
    return a["pos"], a["z"]


def graph(p, z, system, device):
    if system == "ha":
        p = torch.tensor(p - p.mean(0), dtype=torch.float32)
    else:
        p = torch.tensor(p, dtype=torch.float32)
        if system != "dota":
            p -= p.mean(0)
    if system == "dota":
        ei = (~torch.eye(len(p), dtype=torch.bool)).nonzero().T
    else:
        ei = build_knn_edge_index(p, k=24, include_ties=system != "ha")
    d = Data(
        pos=p,
        x=torch.tensor(z, dtype=torch.float32)[:, None],
        edge_index=ei,
        num_nodes=len(p),
    )
    d.edge_attr = torch.linalg.norm(p[ei[0]] - p[ei[1]], dim=1, keepdim=True)
    return d.to(device)


def make_net(groups, device):
    return (
        SymFPWrapper(
            ParaFreeModel(irreps_channels=16, branch_mode="hybrid"),
            point_groups=[canonical_label_pair(g)[1] for g in groups],
            ls_eps=1e-10,
        )
        .eval()
        .to(device)
    )


def spectrum(net, data):
    features = net._resolve_graph_sh(net.encoder(data))
    keys, blocks, _ = merge_irrep_blocks(features, net.hidden_irreps)
    return [block_singular_values(blocks[k].float()) for k in keys]


def streamed_descriptor(points):
    """Same directed-pair coefficients as baseline, bounded temporary memory."""
    p = np.asarray(points, dtype=float)
    total = torch.zeros(49, dtype=torch.float64)
    count = 0
    n = len(p)
    per = 65536
    for start in range(0, n * n, per):
        flat = np.arange(start, min(n * n, start + per))
        raw = p[flat % n] - p[flat // n]
        raw = raw[~np.all(np.isclose(raw, 0, atol=1e-12), axis=1)]
        radii = np.linalg.norm(raw, axis=1)
        raw = raw[radii > 1e-12]
        radii = radii[radii > 1e-12]
        if not len(raw):
            continue
        directions = torch.as_tensor(raw / radii[:, None], dtype=torch.float64)
        values = o3.spherical_harmonics(
            list(range(7)), directions, normalize=True, normalization="integral"
        )
        total += values.sum(0)
        count += len(raw)
    coeff = total / count
    blocks = tuple(coeff[sl].numpy() for sl in engel._harmonic_block_slices(6))
    energies = [float(b @ b) for b in blocks]
    qeq = sum(energies[1:])
    omega = 1 / count
    eee = omega * 48
    return engel.HarmonicDescriptor(
        blocks=blocks,
        raw_total_energy=qeq,
        qeq=qeq,
        omega=omega,
        eee=eee,
        total_order_parameter=qeq / eee - 1,
        per_degree_energy=tuple([0.0] + [e / qeq for e in energies[1:]]),
        num_directions_used=count,
    )


def sample(p, z, system):
    # Group/molecule tables are warmed in setup; this is only coordinate preprocessing.
    symbols = {6: "C", 7: "N", 8: "O", 66: "Dy"}
    centered = np.asarray(p, dtype=float).copy()
    origin = np.zeros(3) if system == "dota" else centered.mean(0)
    centered -= origin
    return PreparedPointCloud(
        system,
        tuple(symbols[int(v)] for v in z),
        centered,
        None,
        origin,
        0,
        len(p),
        len(p),
    )


def build_function(method, system, device, mode="fixed", groups=None):
    groups = groups or ["C6" if system == "p97" else "C3"]
    p, z = data_input(system)
    stages = {}
    if method == "SymFP":
        net = make_net(groups, device)

        def call():
            t = time.perf_counter()
            d = graph(p, z, system, device)
            sync(device)
            t1 = time.perf_counter()
            with torch.inference_mode():
                sv = spectrum(net, d)
                sync(device)
                t2 = time.perf_counter()
                ls, pls = net._compute_soft_scores(sv)
                values = pls[0].cpu().tolist()
            sync(device)
            stages.update(
                graph_transfer_s=t1 - t,
                spectrum_s=t2 - t1,
                target_s=time.perf_counter() - t2,
            )
            return values

        return call, stages, net
    if method == "Engel":
        for g in groups:
            for l in range(1, 7):
                engel._projector_info(g, l)
        # Prewarm e3nn code generation on the complete small real complex, not synthetic inputs.
        small, _ = data_input("dota")
        streamed_descriptor(small)
        original = engel.optimize_orientation

        def traced(*a, **kw):
            t = time.perf_counter()
            try:
                return original(*a, **kw)
            finally:
                stages["search_s"] = stages.get("search_s", 0) + time.perf_counter() - t

        engel.optimize_orientation = traced

        def call():
            stages.clear()
            t = time.perf_counter()
            s = sample(p, z, system)
            desc = streamed_descriptor(s.positions)
            t1 = time.perf_counter()
            original_rotate = engel._rotate_descriptor
            rotated_cache = {}

            def shared_rotate(descriptor, rotation):
                key = np.asarray(rotation).tobytes()
                if key not in rotated_cache:
                    rotated_cache[key] = original_rotate(descriptor, rotation)
                return rotated_cache[key]

            if mode == "fixed":
                engel._rotate_descriptor = shared_rotate
            try:
                vals = [
                    engel.evaluate_candidate(
                        s,
                        desc,
                        g,
                        axis_mode="csom_search" if mode == "search" else "none",
                        num_axis_samples=10,
                        num_angle_samples=2,
                        local_opt_maxiter=100,
                        exclude_identity=True,
                    ).engel_score
                    for g in groups
                ]
            finally:
                engel._rotate_descriptor = original_rotate
            stages.update(
                descriptor_s=t1 - t,
                target_s=time.perf_counter() - t1 - stages.get("search_s", 0),
            )
            return vals

        return call, stages, None
    model = CSoM(
        candidate_groups=groups,
        center_mode="centroid",
        orientation_mode="none",
        num_axis_samples=10,
        num_angle_samples=2,
    )
    assert model.uses_official
    module = model.official_module
    operations = {}
    for g in groups:
        name = "pointgroup_" + g
        names, ops = module.ReadSymmetryFile(name, module.Path_To_Symmetry_Operations)
        operations[g] = (names, np.asarray(ops))

    def call():
        stages.clear()
        s = sample(p, z, system)
        cache = {}
        vals = []
        search_s = 0.0
        score_s = 0.0
        with contextlib.redirect_stdout(io.StringIO()):
            for g in groups:
                names, ops = operations[g]
                q = s.positions
                if mode == "search":
                    t = time.perf_counter()
                    q, *_ = module.find_zaxis(
                        q,
                        np.asarray(s.species, dtype=object),
                        ops,
                        N_input_zaxis=10,
                        N_input_angles=2,
                    )
                    search_s += time.perf_counter() - t
                t = time.perf_counter()
                deviations = []
                for op in ops:
                    key = np.round(op, 12).tobytes()
                    if mode == "search" or key not in cache:
                        operated = q @ op
                        matched, _, _ = module.best_permutation_multiple_atoms(
                            q, np.asarray(s.species, dtype=object), operated
                        )
                        cache[key] = float(
                            module.symmetry_deviation_calculator(matched, q)
                        )
                    deviations.append(cache[key])
                vals.append(float(np.mean(deviations)))
                score_s += time.perf_counter() - t
        stages.update(
            search_s=search_s, fixed_readout_s=score_s, unique_operations=len(cache)
        )
        return vals

    return call, stages, model


def measure(call, stages, device, job, search=False):
    records = []
    peak = 0
    signal.signal(signal.SIGALRM, alarm)

    def trial(role):
        nonlocal peak
        sync(device)
        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        t = time.perf_counter()
        signal.setitimer(signal.ITIMER_REAL, CAP)
        try:
            values = call()
            sync(device)
            elapsed = time.perf_counter() - t
        except TimedOut:
            records.append({"role": role, "status": "timeout", "lower_bound_s": CAP})
            raise
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            if device.startswith("cuda"):
                peak = max(peak, torch.cuda.max_memory_allocated())
        record = {
            "role": role,
            "status": "ok",
            "total_s": elapsed,
            "stages": dict(stages),
            "values": values,
        }
        records.append(record)
        save(OUT / "jobs" / f"{job}.partial.json", records)
        return elapsed

    try:
        if search:
            trial("measured")
        else:
            pilot = trial("calibration")
            warmups, repeats = (3, 10) if pilot <= 2 else (1, 3)
            for _ in range(warmups):
                trial("warmup")
            for _ in range(repeats):
                trial("measured")
        status = "ok"
    except TimedOut:
        status = "timeout"
    measured = [r for r in records if r["role"] == "measured" and r["status"] == "ok"]
    return {
        "job": job,
        "status": status,
        "records": records,
        "median_s": (
            statistics.median(r["total_s"] for r in measured)
            if measured and status == "ok"
            else None
        ),
        "n_measured": len(measured),
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "peak_gpu_allocated_bytes": peak,
        "torch_threads": torch.get_num_threads(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
    }
