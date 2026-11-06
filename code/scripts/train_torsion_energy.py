"""MAE energy pretraining on the frozen torsion train/validation split."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(variable, "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader

from models.backbone_factory import build_backbone
from utils.torsion_models import MODELS, baseline_dir, initialization, model_sources
from scripts.torsion_pretraining import trajectory as read_trajectory
from utils.torsion_pretraining_analysis import contained_path
from utils.torsion_data import prepare_data, save_protocol

ATOMIC_NUMBERS = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9, "S": 16, "Cl": 17, "Br": 35}
from utils.toy_runtime_utils import build_node_features, build_distance_edge_attr
from utils.torsion_training_log import init_run, log_epoch


def write_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class EnergyModel(nn.Module):
    def __init__(self, initial, device):
        super().__init__()
        self.backbone = build_backbone(**initial["config"], device=device)
        self.backbone.load_state_dict(initial["state_dict"], strict=True)
        width = initial["config"]["hidden_dim"]
        self.energy_head = nn.Sequential(
            nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 1)
        )
        self.to(device)

    def forward(self, data):
        return self.energy_head(self.backbone(data)["graph_feat"]).flatten()


def load_data(data_dir, model_type="equiformer", protocol_out=None):
    frames, _, split, norm = prepare_data(data_dir)
    if protocol_out is not None:
        save_protocol(protocol_out, split, norm)
    assert frames.record_id.nunique() == 86 and len(frames) == 2064
    datasets = {"train": [], "validation": []}
    for record_id, group in frames.groupby("record_id", sort=True):
        assert group.split.nunique() == 1
        part = group.split.iloc[0]
        assert record_id in split[f"{part}_record_ids"]
        trajectory = read_trajectory(
            contained_path(data_dir, group.trajectory_path.iloc[0])
        )
        assert len(trajectory) == len(group) == 24
        for row in group.itertuples():
            symbols, coords = trajectory[row.frame_index]
            atomic = torch.tensor(
                [ATOMIC_NUMBERS[s] for s in symbols], dtype=torch.long
            )
            pos = torch.tensor(coords, dtype=torch.float64)
            pos = (pos - pos.mean(0, keepdim=True)).float()
            grid = torch.arange(len(pos))
            i, j = torch.meshgrid(grid, grid, indexing="ij")
            mask = i != j
            edges = torch.stack([i[mask], j[mask]])
            expected = (row.energy_hartree - norm["mean_hartree"]) / norm["std_hartree"]
            assert abs(row.energy_normalized - expected) < 1e-12
            data = Data(
                x=build_node_features(atomic, model_type),
                pos=pos,
                edge_index=edges,
                num_nodes=len(pos),
                atomic_numbers=atomic,
                y=torch.tensor([row.energy_normalized], dtype=torch.float32),
                record_id=torch.tensor([record_id]),
                frame_index=torch.tensor([row.frame_index]),
            )
            data.edge_attr = build_distance_edge_attr(pos, edges)
            datasets[part].append(data)
    assert len(datasets["train"]) == 1656 and len(datasets["validation"]) == 408
    return datasets, split, norm


def save_checkpoint(path, model, optimizer, scheduler, generator, **metadata):
    state = dict(
        model_state_dict=model.state_dict(),
        optimizer_state_dict=optimizer.state_dict(),
        scheduler_state_dict=scheduler.state_dict(),
        loader_rng=generator.get_state(),
        torch_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state_all(),
        numpy_rng=np.random.get_state(),
        python_rng=random.getstate(),
        **metadata,
    )
    temp = path.with_suffix(".tmp")
    torch.save(state, temp)
    temp.replace(path)


def evaluate(model, loader, device):
    model.eval()
    absolute, count = 0.0, 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            prediction = model(data)
            if not torch.isfinite(prediction).all():
                raise FloatingPointError("Nonfinite validation prediction")
            absolute += (prediction - data.y).abs().double().sum().item()
            count += data.num_graphs
    return absolute / count


def smoke(model, datasets, device, out, batch_size):
    """Check graph batching, O(3) invariance and an actual MAE optimization step."""
    from e3nn import o3

    examples = datasets["train"][::24][:batch_size]
    batch = Batch.from_data_list(examples).to(device)
    model.eval()
    with torch.no_grad():
        together = model(batch)
        single = torch.cat(
            [model(Batch.from_data_list([d]).to(device)) for d in examples]
        )
        rotated = batch.clone()
        rotated.pos = rotated.pos @ (-o3.rand_matrix(device=device)).T
        changed = model(rotated)
    assert torch.allclose(together, single, atol=1e-5, rtol=1e-4)
    assert torch.allclose(together, changed, atol=1e-5, rtol=1e-4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    model.train()
    loss = nn.functional.l1_loss(model(batch), batch.y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    backbone_grad = sum(
        float(p.grad.square().sum())
        for p in model.backbone.parameters()
        if p.grad is not None
    )
    assert np.isfinite(backbone_grad) and backbone_grad > 0
    before = {k: p.detach().clone() for k, p in model.backbone.named_parameters()}
    norm = nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
    optimizer.step()
    changed_parameters = [
        k for k, p in model.backbone.named_parameters() if not torch.equal(before[k], p)
    ]
    assert changed_parameters
    # Exercise the same spectral wrapper used for retrieval after the update.
    from models.symfp_wrapper import SymFPWrapper
    from utils.torsion_pretraining_analysis import (
        GROUPS as EVALUATION_GROUPS,
        HM as SCHOENFLIES_TO_HM,
    )

    wrapper = (
        SymFPWrapper(
            model.backbone,
            ls_eps=1e-10,
            point_groups=[SCHOENFLIES_TO_HM[g] for g in EVALUATION_GROUPS],
        )
        .to(device)
        .eval()
    )
    with torch.no_grad():
        spectral = wrapper(Batch.from_data_list([examples[0]]).to(device))
    assert all(torch.isfinite(v).all() for v in spectral["block_singular_values"])
    generator = torch.Generator().manual_seed(42)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    save_checkpoint(
        out / "smoke_checkpoint.pt",
        model,
        optimizer,
        scheduler,
        generator,
        epoch=1,
        model_type=model.backbone.__class__.__name__,
    )
    checkpoint = torch.load(out / "smoke_checkpoint.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    assert all(
        torch.equal(v, checkpoint["model_state_dict"][k])
        for k, v in model.state_dict().items()
    )
    valid = evaluate(
        model,
        DataLoader(datasets["validation"][:batch_size], batch_size=batch_size),
        device,
    )
    report = dict(
        status="passed",
        model=model.backbone.__class__.__name__,
        checkpoint_roundtrip="passed",
        spectral_wrapper="finite singular values",
        distinct_molecules=len(examples),
        batch_size=batch_size,
        mae=float(loss),
        validation_batch_mae=valid,
        gradient_norm=float(norm),
        backbone_gradient_squared_norm=backbone_grad,
        batch_max_difference=float((together - single).abs().max()),
        rotation_reflection_max_difference=float((together - changed).abs().max()),
        changed_backbone_parameters=changed_parameters,
        peak_gpu_memory_gb=(
            torch.cuda.max_memory_allocated(device) / 1e9
            if device.type == "cuda"
            else 0
        ),
        note="Independent process; production training reloads original initialization",
    )
    write_json(out / "smoke.json", report)
    print(json.dumps(report), flush=True)


def main(args):
    args.out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    datasets, split, normalization = load_data(args.data, args.model, args.out)
    initial = initialization(args.initial, args.model, args.seed, device)
    assert initial["seed"] == args.seed == split["seed"] == 42
    model = EnergyModel(initial, device)
    if args.smoke:
        smoke(model, datasets, device, args.out, args.batch_size)
        return
    if (args.out / "last.pt").exists() and not args.resume:
        raise FileExistsError(
            "Existing training checkpoint: use --resume or another output directory"
        )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        datasets["train"],
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    valid_loader = DataLoader(
        datasets["validation"], batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=9, threshold=0.0, min_lr=1e-6
    )
    paths = [
        args.initial,
        args.data / "responses.csv",
        args.out / "split.json",
        args.out / "normalization.json",
        ROOT / "utils/torsion_data.py",
        Path(__file__),
        ROOT / "utils/torsion_training_log.py",
    ] + model_sources(args.model)
    config = dict(
        **initial["config"],
        seed=args.seed,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        optimizer="AdamW",
        weight_decay=1e-5,
        max_epochs=args.epochs,
        loss="MAE",
        target="z-score normalized total energy_hartree",
        normalization=normalization,
        gradient_clip_norm=5.0,
        lr_plateau_bad_epochs=10,
        lr_factor=0.5,
        early_stopping_bad_epochs=30,
        checkpoint_metric="validation/mae_normalized",
        counts=split["counts"],
        split_seed=split["seed"],
        pooling="existing graph_feat mean pooling",
        head="Linear(64,64), SiLU, Linear(64,1)",
        device=str(device),
        gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        precision="float32; TF32 disabled; no AMP",
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        source_sha256={
            os.path.relpath(p.resolve(), ROOT.parent): sha(p) for p in paths
        },
    )
    epoch, best_epoch, bad_epochs, best = 0, 0, 0, float("inf")
    if args.resume:
        previous_config = json.loads((args.out / "config.json").read_text())
        assert previous_config == config, "Training configuration or inputs changed"
        state = torch.load(args.out / "last.pt", map_location=device)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        generator.set_state(state["loader_rng"].cpu())
        torch.set_rng_state(state["torch_rng"].cpu())
        torch.cuda.set_rng_state_all([v.cpu() for v in state["cuda_rng"]])
        np.random.set_state(state["numpy_rng"])
        random.setstate(state["python_rng"])
        epoch, best_epoch, bad_epochs, best = (
            state[k] for k in ["epoch", "best_epoch", "bad_epochs", "best_mae"]
        )
    else:
        write_json(args.out / "config.json", config)
    run = init_run(args.out, config, resume=args.resume)
    from utils import torsion_training_log as logger

    started = time.time()
    print(f"Local log: {run.url}", flush=True)
    print(
        f"Training {len(datasets['train'])} / validation {len(datasets['validation'])}; device={device}",
        flush=True,
    )
    try:
        if epoch == 0:
            initial_mae = evaluate(model, valid_loader, device)
            logger.log(
                {
                    "validation/initial_mae_normalized": initial_mae,
                    "validation/initial_mae_hartree": initial_mae
                    * normalization["std_hartree"],
                },
                step=0,
            )
            write_json(
                args.out / "initial_validation.json", {"mae_normalized": initial_mae}
            )
        while epoch < args.epochs and bad_epochs < 30:
            epoch += 1
            tick = time.time()
            model.train()
            total, count = 0.0, 0
            lr_used = optimizer.param_groups[0]["lr"]
            for batch_index, data in enumerate(train_loader, 1):
                data = data.to(device)
                optimizer.zero_grad(set_to_none=True)
                prediction = model(data)
                loss = nn.functional.l1_loss(prediction, data.y)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite training loss")
                loss.backward()
                nn.utils.clip_grad_norm_(
                    model.parameters(), 5.0, error_if_nonfinite=True
                )
                optimizer.step()
                total += float(loss.detach()) * data.num_graphs
                count += data.num_graphs
                if batch_index == 1 or batch_index % 50 == 0:
                    write_json(
                        args.out / "status.json",
                        dict(
                            state="running",
                            pid=os.getpid(),
                            epoch=epoch,
                            batch=batch_index,
                            batches=len(train_loader),
                            best_epoch=best_epoch,
                            best_mae=None if not np.isfinite(best) else best,
                            elapsed_seconds=time.time() - started,
                        ),
                    )
                    print(
                        f"epoch {epoch} batch {batch_index}/{len(train_loader)} train_mae={total/count:.6f}",
                        flush=True,
                    )
            train_mae = total / count
            validation_mae = evaluate(model, valid_loader, device)
            improved = validation_mae < best
            if improved:
                best, best_epoch, bad_epochs = validation_mae, epoch, 0
            else:
                bad_epochs += 1
            scheduler.step(validation_mae)
            metadata = dict(
                epoch=epoch,
                best_epoch=best_epoch,
                best_mae=best,
                bad_epochs=bad_epochs,
                config=config,
            )
            save_checkpoint(
                args.out / "last.pt", model, optimizer, scheduler, generator, **metadata
            )
            if improved:
                save_checkpoint(
                    args.out / "best.pt",
                    model,
                    optimizer,
                    scheduler,
                    generator,
                    **metadata,
                )
            row = dict(
                epoch=epoch,
                train_mae_normalized=train_mae,
                validation_mae_normalized=validation_mae,
                train_mae_hartree=train_mae * normalization["std_hartree"],
                validation_mae_hartree=validation_mae * normalization["std_hartree"],
                best_validation_mae=best,
                best_epoch=best_epoch,
                learning_rate=lr_used,
                seconds=time.time() - tick,
            )
            path = args.out / "metrics.csv"
            with path.open("a", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                if handle.tell() == 0:
                    writer.writeheader()
                writer.writerow(row)
            log_epoch(
                epoch=epoch,
                train_mae=train_mae,
                validation_mae=validation_mae,
                energy_std_hartree=normalization["std_hartree"],
                learning_rate=lr_used,
                best_validation_mae=best,
            )
            print(json.dumps(row), flush=True)
        write_json(
            args.out / "status.json",
            dict(
                state="complete",
                epoch=epoch,
                best_epoch=best_epoch,
                best_validation_mae_normalized=best,
                best_validation_mae_hartree=best * normalization["std_hartree"],
                stop_reason="early_stopping" if bad_epochs >= 30 else "max_epochs",
                elapsed_seconds=time.time() - started,
            ),
        )
    except BaseException as exc:
        write_json(
            args.out / "status.json",
            dict(state="failed", epoch=epoch, error_type=type(exc).__name__),
        )
        raise
    finally:
        logger.finish()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=MODELS, default="equiformer")
    p.add_argument("--data", type=Path, default=ROOT.parent / "datasets/torsion")
    p.add_argument("--initial", type=Path, default=None)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    args.initial = args.initial or baseline_dir(args.model) / f"{args.model}_random.pt"
    args.out = (
        args.out or ROOT.parent / f"outputs/torsion_energy/{args.model}_seed{args.seed}"
    )
    main(args)
