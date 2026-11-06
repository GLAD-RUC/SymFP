"""Uni-Mol-style masked atom / coordinate / distance pretraining of four GNNs.

Adaptation: existing all-atom graphs and equivariant node-position outputs; no
Transformer pair-state regularizers or BOS/EOS nodes. Same frozen torsion split.
"""

from __future__ import annotations
import argparse
import csv
import json
import os
from pathlib import Path
import random
import sys
import time

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(variable, "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from models.backbone_factory import build_backbone
from scripts.train_torsion_energy import load_data, write_json, sha, save_checkpoint
from utils.torsion_models import MODELS, baseline_dir, initialization, model_sources
from utils.torsion_training_log import init_run
from utils import torsion_training_log as logger

# Published Uni-Mol molecular pretraining distance-target normalization constants.
DIST_MEAN, DIST_STD = 6.312581655060595, 3.3899264663911888
BASE = ROOT.parent / "outputs/torsion_unimol"


class CorruptedConformers(torch.utils.data.Dataset):
    def __init__(self, graphs, vocabulary, training, seed=42):
        self.graphs, self.vocabulary = graphs, np.asarray(vocabulary)
        self.training, self.seed, self.epoch = training, seed, 0

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, index):
        clean = self.graphs[index]
        # Identical corruption across architectures; validation is fixed across epochs.
        rng = np.random.default_rng(
            np.random.SeedSequence(
                [
                    self.seed,
                    self.epoch if self.training else 0,
                    int(clean.record_id),
                    int(clean.frame_index),
                    0 if self.training else 1,
                ]
            )
        )
        data = clean.clone()
        n = data.num_nodes
        selected = rng.choice(n, max(1, int(0.15 * n + rng.random())), replace=False)
        mask = torch.zeros(n, dtype=torch.bool)
        mask[selected] = True
        decision = rng.random(len(selected))
        unchanged = decision < 0.05
        random_replacement = (decision >= 0.05) & (decision < 0.10)
        modified = selected[~unchanged]
        tokens = clean.atomic_numbers.clone()
        tokens[modified] = 0  # Unused element-0 one-hot slot represents [MASK].
        random_indices = selected[random_replacement]
        tokens[random_indices] = torch.tensor(
            rng.choice(self.vocabulary, len(random_indices))
        )
        data.x = F.one_hot(tokens, num_classes=101).float()
        data.atomic_numbers = tokens
        data.pos[modified] += torch.tensor(
            rng.uniform(-1.0, 1.0, (len(modified), 3)), dtype=torch.float32
        )
        src, dst = data.edge_index
        data.edge_attr = (data.pos[src] - data.pos[dst]).norm(dim=-1, keepdim=True)
        data.clean_pos = clean.pos.clone()
        data.atom_target = clean.atomic_numbers.clone()
        data.masked_atoms = mask
        # Energy is deliberately absent from the self-supervised inputs and loss.
        del data.y
        return data


class DenoisingModel(nn.Module):
    def __init__(self, initial, device):
        super().__init__()
        self.backbone = build_backbone(**initial["config"], device=device)
        self.backbone.load_state_dict(initial["state_dict"], strict=True)
        width = initial["config"]["hidden_dim"]
        self.atom_head = nn.Sequential(
            nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 101)
        )
        # Symmetric invariant pair features give the same prediction for i,j and j,i.
        self.distance_head = nn.Sequential(
            nn.Linear(2 * width + 1, width), nn.SiLU(), nn.Linear(width, 1)
        )
        self.to(device)

    def forward(self, data):
        output = self.backbone(data)
        features = output["node_feat"]
        same_graph = data.batch[:, None] == data.batch[None, :]
        src, dst = torch.where(data.masked_atoms[:, None] & same_graph)
        noisy_distance = (data.pos[src] - data.pos[dst]).norm(dim=-1, keepdim=True)
        pair = torch.cat(
            [
                features[src] + features[dst],
                (features[src] - features[dst]).abs(),
                noisy_distance,
            ],
            -1,
        )
        return dict(
            atom_logits=self.atom_head(features[data.masked_atoms]),
            coordinates=output["node_pos"],
            distances=self.distance_head(pair).flatten(),
            pair_src=src,
            pair_dst=dst,
        )


def objective(prediction, data):
    token = F.cross_entropy(
        prediction["atom_logits"], data.atom_target[data.masked_atoms]
    )
    coord = F.smooth_l1_loss(
        prediction["coordinates"][data.masked_atoms],
        data.clean_pos[data.masked_atoms],
        beta=1.0,
    )
    src, dst = prediction["pair_src"], prediction["pair_dst"]
    target = (
        (data.clean_pos[src] - data.clean_pos[dst]).norm(dim=-1) - DIST_MEAN
    ) / DIST_STD
    distance = F.smooth_l1_loss(prediction["distances"], target, beta=1.0)
    total = token + 5 * coord + 10 * distance
    accuracy = (
        (prediction["atom_logits"].argmax(-1) == data.atom_target[data.masked_atoms])
        .float()
        .mean()
    )
    return total, dict(
        total=float(total.detach()),
        atom=float(token.detach()),
        coordinate=float(coord.detach()),
        distance=float(distance.detach()),
        atom_accuracy=float(accuracy.detach()),
    )


def epoch_pass(model, loader, device, optimizer=None, progress=None):
    model.train(optimizer is not None)
    sums, count = {}, 0
    with torch.set_grad_enabled(optimizer is not None):
        for step, data in enumerate(loader, 1):
            data = data.to(device)
            loss, metrics = objective(model(data), data)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite denoising loss")
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    model.parameters(), 5.0, error_if_nonfinite=True
                )
                optimizer.step()
            # Fixed batch composition in validation; same aggregation across models.
            for key, value in metrics.items():
                sums[key] = sums.get(key, 0.0) + value * data.num_graphs
            count += data.num_graphs
            if progress and (step == 1 or step % 50 == 0):
                progress(step, len(loader))
    return {key: value / count for key, value in sums.items()}


def smoke(model, datasets, device, out, initial):
    from e3nn import o3
    from models.symfp_wrapper import SymFPWrapper
    from utils.torsion_pretraining_analysis import (
        GROUPS as EVALUATION_GROUPS,
        HM as SCHOENFLIES_TO_HM,
    )

    dataset = datasets["train"]
    indices = list(range(0, 24 * 8, 24))
    examples = [dataset[i] for i in indices]
    assert all(torch.equal(dataset[i].pos, dataset[i].pos) for i in indices)
    batch = Batch.from_data_list(examples).to(device)
    assert torch.equal(batch.x.argmax(-1), batch.atomic_numbers)
    assert all(not hasattr(g, "y") or g.y is None for g in examples)
    src, dst = batch.edge_index
    assert torch.allclose(
        batch.edge_attr.flatten(), (batch.pos[src] - batch.pos[dst]).norm(dim=-1)
    )
    # Changing training epoch changes corruption; validation remains identical.
    original = dataset[0].pos.clone()
    dataset.epoch += 1
    assert not torch.equal(original, dataset[0].pos)
    dataset.epoch -= 1
    valid = datasets["validation"]
    v = valid[0].pos.clone()
    valid.epoch += 1
    assert torch.equal(v, valid[0].pos)
    model.eval()
    with torch.no_grad():
        together = model(batch)
        singles = [model(Batch.from_data_list([g]).to(device)) for g in examples]
        assert torch.allclose(
            together["atom_logits"],
            torch.cat([p["atom_logits"] for p in singles]),
            atol=2e-5,
            rtol=1e-4,
        )
        assert torch.allclose(
            together["coordinates"],
            torch.cat([p["coordinates"] for p in singles]),
            atol=2e-5,
            rtol=1e-4,
        )
        assert torch.allclose(
            together["distances"],
            torch.cat([p["distances"] for p in singles]),
            atol=2e-5,
            rtol=1e-4,
        )
        rotated = batch.clone()
        rotation = -o3.rand_matrix(device=device)
        rotated.pos = rotated.pos @ rotation.T
        pred = model(rotated)
        coord_error = float(
            (pred["coordinates"] - together["coordinates"] @ rotation.T).abs().max()
        )
        assert torch.allclose(
            pred["coordinates"],
            together["coordinates"] @ rotation.T,
            atol=2e-5,
            rtol=1e-4,
        )
        assert torch.allclose(
            pred["distances"], together["distances"], atol=2e-5, rtol=1e-4
        )
        assert torch.allclose(
            pred["atom_logits"], together["atom_logits"], atol=2e-5, rtol=1e-4
        )
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    before = {k: p.detach().clone() for k, p in model.backbone.named_parameters()}
    metrics = epoch_pass(model, DataLoader(examples, batch_size=8), device, optimizer)
    changed = [
        k for k, p in model.backbone.named_parameters() if not torch.equal(before[k], p)
    ]
    assert changed
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.atom_head.parameters()
    )
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.distance_head.parameters()
    )
    torch.save(
        dict(model_state_dict=model.state_dict(), config=initial["config"]),
        out / "smoke_checkpoint.pt",
    )
    state = torch.load(out / "smoke_checkpoint.pt", map_location=device)
    model.load_state_dict(state["model_state_dict"], strict=True)
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
        result = wrapper(Batch.from_data_list([dataset.graphs[0]]).to(device))
    assert all(torch.isfinite(v).all() for v in result["block_singular_values"])
    report = dict(
        status="passed",
        model=initial["config"]["model_type"],
        loss_components=metrics,
        coordinate_equivariance_max_error=coord_error,
        changed_backbone_parameters=changed,
        peak_gpu_memory_gb=(
            torch.cuda.max_memory_allocated(device) / 1e9
            if device.type == "cuda"
            else 0
        ),
        corruption_reproducibility="passed",
        energy_excluded=True,
        batching="passed",
        checkpoint="passed",
        clean_pls="passed",
    )
    write_json(out / "smoke.json", report)
    print(json.dumps(report), flush=True)


def main(args):
    args.out.mkdir(parents=True, exist_ok=True)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    graphs, split, _ = load_data(args.data, args.model, args.out)
    vocabulary = sorted(
        set(torch.cat([d.atomic_numbers for d in graphs["train"]]).tolist())
    )
    datasets = {
        key: CorruptedConformers(value, vocabulary, key == "train")
        for key, value in graphs.items()
    }
    initial_path = args.initial or baseline_dir(args.model) / f"{args.model}_random.pt"
    initial = initialization(initial_path, args.model, 42, device)
    model = DenoisingModel(initial, device)
    if args.smoke:
        smoke(model, datasets, device, args.out, initial)
        return
    if (args.out / "last.pt").exists() and not args.resume:
        raise FileExistsError("Existing run; use --resume")
    generator = torch.Generator().manual_seed(42)
    train_loader = DataLoader(
        datasets["train"], batch_size=8, shuffle=True, generator=generator
    )
    valid_loader = DataLoader(datasets["validation"], batch_size=8, shuffle=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=9, threshold=0.0, min_lr=1e-6
    )
    sources = [
        Path(__file__),
        ROOT / "scripts/train_torsion_energy.py",
        ROOT / "utils/torsion_training_log.py",
        initial_path,
        args.out / "split.json",
        args.data / "responses.csv",
        ROOT / "utils/torsion_data.py",
    ] + model_sources(args.model)
    config = dict(
        **initial["config"],
        seed=42,
        split=split["counts"],
        batch_size=8,
        max_epochs=args.epochs,
        objective="Uni-Mol-style atom CE + 5 coordinate SmoothL1 + 10 distance SmoothL1",
        loss_weights=[1, 5, 10],
        beta=1.0,
        mask_probability=0.15,
        leave_unmasked_probability=0.05,
        random_token_probability=0.05,
        noise="uniform [-1,1] Angstrom, modified selected atoms only",
        mask_slot=0,
        random_token_vocabulary=vocabulary,
        distance_target_mean=DIST_MEAN,
        distance_target_std=DIST_STD,
        validation_corruption="fixed seed42 per molecule/frame; epoch0",
        checkpoint_metric="validation/total",
        learning_rate=3e-4,
        optimizer="AdamW",
        weight_decay=1e-5,
        lr_plateau_bad_epochs=10,
        early_stopping_bad_epochs=30,
        adaptations=[
            "all hydrogens retained",
            "no BOS/EOS",
            "existing equivariant node_pos readout",
            "symmetric node-pair distance head",
            "no Transformer norm regularizers",
            "uniform random replacements from training elements",
            "at least one selected atom",
        ],
        reference="https://github.com/deepmodeling/Uni-Mol/tree/90f52c41299a1a582da0f9765e9f87aa21faa16a/unimol",
        reference_commit="90f52c41299a1a582da0f9765e9f87aa21faa16a",
        source_sha256={
            os.path.relpath(p.resolve(), ROOT.parent): sha(p) for p in sources
        },
    )
    epoch, best_epoch, bad_epochs, best = 0, 0, 0, float("inf")
    if args.resume:
        assert json.loads((args.out / "config.json").read_text()) == config
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
    init_run(args.out, config, resume=args.resume)
    start = time.time()
    try:
        while epoch < args.epochs and bad_epochs < 30:
            epoch += 1
            datasets["train"].epoch = epoch

            def progress(step, total):
                write_json(
                    args.out / "status.json",
                    dict(
                        state="running",
                        epoch=epoch,
                        batch=step,
                        batches=total,
                        best_epoch=best_epoch,
                        best_loss=None if not np.isfinite(best) else best,
                        pid=os.getpid(),
                    ),
                )
                print(f"epoch {epoch} batch {step}/{total}", flush=True)

            training = epoch_pass(model, train_loader, device, optimizer, progress)
            validation = epoch_pass(model, valid_loader, device)
            improved = validation["total"] < best
            if improved:
                best, best_epoch, bad_epochs = validation["total"], epoch, 0
            else:
                bad_epochs += 1
            lr_used = optimizer.param_groups[0]["lr"]
            scheduler.step(validation["total"])
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
                **{f"train_{k}": v for k, v in training.items()},
                **{f"validation_{k}": v for k, v in validation.items()},
                learning_rate=lr_used,
                best_epoch=best_epoch,
            )
            with (args.out / "metrics.csv").open("a", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                if handle.tell() == 0:
                    writer.writeheader()
                writer.writerow(row)
            logger.log(
                {
                    **{f"train/{k}": v for k, v in training.items()},
                    **{f"validation/{k}": v for k, v in validation.items()},
                    "validation/best_total": best,
                    "optimizer/learning_rate": lr_used,
                },
                step=epoch,
            )
            print(json.dumps(row), flush=True)
        write_json(
            args.out / "status.json",
            dict(
                state="complete",
                epoch=epoch,
                best_epoch=best_epoch,
                best_validation_loss=best,
                stop_reason="early_stopping" if bad_epochs >= 30 else "max_epochs",
                elapsed_seconds=time.time() - start,
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
    p.add_argument("--model", choices=MODELS, required=True)
    p.add_argument("--out", type=Path)
    p.add_argument("--data", type=Path, default=ROOT.parent / "datasets/torsion")
    p.add_argument("--initial", type=Path)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    args.out = args.out or BASE / f"{args.model}_seed42"
    main(args)
