from __future__ import annotations

from pathlib import Path
from typing import Any
import warnings

import torch

from .equiformer import EquiformerModel
from .hegnn import HEGNNModel
from .mace import MACEModel
from .para_free import ParaFreeModel
from .tfn import TFNModel


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TOY_CKPT_DIR = PROJECT_ROOT.parent / "checkpoints/toy"
SUPPORTED_BACKBONES = ("para_free", "hegnn", "tfn", "mace", "equiformer")
ALLOWED_AUXILIARY_MISSING_PREFIXES = ("node_scalar_readout", "node_pos_proj")


def build_backbone(
    model_type: str,
    *,
    hidden_dim: int = 64,
    num_layers: int = 2,
    max_ell: int = 6,
    irreps_channels: int = 16,
    mace_correlation: int = 2,
    pf_component_mode: str = "none",
    pf_branch_mode: str = "hybrid",
    pf_body_local_scale: float = 1.0,
    pf_water_local_scale: float = 1.0,
    pf_relative_vector_scale: float = 1.0,
    grid_resolution: int | None = None,
    device: str | torch.device = "cpu",
):
    model_type = str(model_type)
    if model_type not in SUPPORTED_BACKBONES:
        raise ValueError(f"Unsupported backbone: {model_type}")

    common_kwargs: dict[str, Any] = {
        "max_ell": int(max_ell),
        "hidden_dim": int(hidden_dim),
        "irreps_channels": int(irreps_channels),
        "device": str(device),
    }

    if model_type == "para_free":
        return ParaFreeModel(
            max_ell=int(max_ell),
            irreps_channels=int(irreps_channels),
            node_input_dim=1,
            edge_input_dim=1,
            component_mode=str(pf_component_mode),
            branch_mode=str(pf_branch_mode),
            body_local_scale=float(pf_body_local_scale),
            water_local_scale=float(pf_water_local_scale),
            relative_vector_scale=float(pf_relative_vector_scale),
            device=str(device),
        )



    if model_type == "hegnn":
        return HEGNNModel(
            num_layer=int(num_layers),
            node_input_dim=101,
            edge_input_dim=1,
            **common_kwargs,
        )

    if model_type == "tfn":
        return TFNModel(
            num_layer=int(num_layers),
            node_input_dim=101,
            edge_input_dim=1,
            **common_kwargs,
        )

    if model_type == "mace":
        return MACEModel(
            correlation=int(mace_correlation),
            num_layer=int(num_layers),
            node_input_dim=101,
            edge_input_dim=1,
            **common_kwargs,
        )

    if model_type == "equiformer":
        return EquiformerModel(
            num_layer=int(num_layers),
            node_input_dim=101,
            edge_input_dim=1,
            **common_kwargs,
        )

    raise AssertionError(f"Unhandled backbone type: {model_type}")


def load_checkpoint_metadata(ckpt_path: str | Path) -> dict[str, Any]:
    from utils.toy_model_utils import load_checkpoint_payload
    checkpoint = load_checkpoint_payload(ckpt_path)
    return dict(checkpoint.get("hyper_parameters", {}))


def load_backbone_weights_from_pretrain(backbone, ckpt_path: str | Path) -> dict[str, list[str]]:
    from utils.toy_model_utils import load_checkpoint_payload
    checkpoint = load_checkpoint_payload(ckpt_path)
    state_dict = checkpoint.get("state_dict", {})
    backbone_state = {
        key[len("backbone.") :]: value
        for key, value in state_dict.items()
        if key.startswith("backbone.")
    }
    if not backbone_state:
        raise ValueError(f"No backbone-prefixed weights found in checkpoint: {ckpt_path}")

    missing_keys, unexpected_keys = backbone.load_state_dict(backbone_state, strict=False)
    allowed_missing_keys = [
        key
        for key in missing_keys
        if any(key.startswith(prefix) for prefix in ALLOWED_AUXILIARY_MISSING_PREFIXES)
    ]
    disallowed_missing_keys = [key for key in missing_keys if key not in allowed_missing_keys]
    if disallowed_missing_keys or unexpected_keys:
        raise RuntimeError(
            "Failed to load pretrained backbone cleanly. "
            f"Missing keys: {missing_keys}. Unexpected keys: {unexpected_keys}."
        )
    if allowed_missing_keys:
        warnings.warn(
            "Loaded backbone checkpoint with missing auxiliary heads that are not used by the "
            f"current graph-level SymFP probe: {allowed_missing_keys}",
            stacklevel=2,
        )
    return {
        "allowed_missing_keys": allowed_missing_keys,
        "unexpected_keys": list(unexpected_keys),
    }


def default_pretrained_ckpt(model_type: str, max_ell: int = 6) -> Path:
    model_type = str(model_type)
    if model_type == "para_free":
        raise ValueError("Parameter-free backbone does not use pretrained checkpoints.")
    if int(max_ell) != 6:
        raise ValueError("Distributed pretrained weights use degree six; pass an explicit checkpoint for another degree.")
    ckpt_path = DEFAULT_TOY_CKPT_DIR / f"{model_type}_pretrained.npz"
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Missing default pretrained checkpoint: {ckpt_path}")
    return ckpt_path


def build_backbone_from_pretrained(
    model_type: str,
    *,
    ckpt_path: str | Path | None = None,
    device: str | torch.device = "cpu",
):
    if model_type == "para_free":
        return build_backbone(model_type="para_free", device=device), {
            "model_type": "para_free",
            "init_mode": "parameter_free",
            "checkpoint_path": None,
            "hidden_dim": 0,
            "num_layers": 0,
            "max_ell": 6,
            "irreps_channels": 16,
            "mace_correlation": 2,
        }

    resolved_ckpt = Path(ckpt_path) if ckpt_path is not None else default_pretrained_ckpt(model_type)
    metadata = load_checkpoint_metadata(resolved_ckpt)
    model = build_backbone(
        model_type=model_type,
        hidden_dim=int(metadata.get("hidden_dim", 64)),
        num_layers=int(metadata.get("num_layers", 2)),
        max_ell=int(metadata.get("max_ell", 6)),
        irreps_channels=int(metadata.get("irreps_channels", 16)),
        mace_correlation=int(metadata.get("mace_correlation", 2)),
        device=device,
    )
    load_report = load_backbone_weights_from_pretrain(model, resolved_ckpt)
    metadata = {
        "model_type": model_type,
        "init_mode": "pretrained",
        "checkpoint_path": str(resolved_ckpt),
        "allowed_missing_keys": load_report["allowed_missing_keys"],
        "unexpected_keys": load_report["unexpected_keys"],
        "hidden_dim": int(metadata.get("hidden_dim", 64)),
        "num_layers": int(metadata.get("num_layers", 2)),
        "max_ell": int(metadata.get("max_ell", 6)),
        "irreps_channels": int(metadata.get("irreps_channels", 16)),
        "mace_correlation": int(metadata.get("mace_correlation", 2)),
    }
    return model, metadata


__all__ = [
    "DEFAULT_TOY_CKPT_DIR",
    "SUPPORTED_BACKBONES",
    "build_backbone",
    "build_backbone_from_pretrained",
    "default_pretrained_ckpt",
    "load_backbone_weights_from_pretrain",
    "load_checkpoint_metadata",
]
