"""Shared model identities and frozen initialization for torsion experiments."""

from pathlib import Path
import torch
from models.backbone_factory import build_backbone

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("equiformer", "tfn", "mace", "hegnn")
NAMES = dict(equiformer="Equiformer", tfn="TFN", mace="MACE", hegnn="HEGNN")


def baseline_dir(model):
    return ROOT.parent / "checkpoints/torsion"


def model_sources(model):
    paths = [
        ROOT / f"models/{model}.py",
        ROOT / "models/backbone_factory.py",
        Path(__file__),
    ]
    if model == "equiformer":
        paths.append(ROOT / "models/equiformer_modules.py")
    paths.append(ROOT / "models/utils.py")
    if model in ("tfn", "mace"):
        paths.extend(sorted((ROOT / "models/mace_modules").glob("*.py")))
    return paths


def initialization(path, model, seed=42, device="cpu"):
    path = Path(path)
    config = dict(
        model_type=model, hidden_dim=64, num_layers=2, max_ell=6, irreps_channels=16
    )
    if model == "mace":
        config["mace_correlation"] = 2
    if path.exists():
        initial = torch.load(path, map_location="cpu", weights_only=True)
        if "backbone_state_dict" in initial:
            if initial["model_type"] != model or initial["variant"] != "random":
                raise ValueError("Training requires the paired random initialization")
            initial = dict(
                seed=initial["seed"],
                config=initial["backbone_config"],
                state_dict=initial["backbone_state_dict"],
            )
        if initial["seed"] != seed or initial["config"] != config:
            raise ValueError(f"Initialization configuration mismatch: {path}")
        return initial
    # Keep energy-head initialization independent of whether this file already exists.
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        backbone = build_backbone(**config, device=device)
        initial = dict(
            config=config,
            seed=seed,
            state_dict={k: v.detach().cpu() for k, v in backbone.state_dict().items()},
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation protects the paired baseline from accidental replacement.
    with path.open("xb") as handle:
        torch.save(initial, handle)
    return initial
