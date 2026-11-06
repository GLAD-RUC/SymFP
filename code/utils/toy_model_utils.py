from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from models.equiformer import EquiformerModel
from models.equiformer_balance import EquiformerBalanceModel
from models.hegnn import HEGNNModel
from models.hegnn_balance import HEGNNBalanceModel
from models.mace import MACEModel
from models.mace_balance import MACEBalanceModel
from models.para_free import ParaFreeModel
from models.tfn import TFNModel
from models.tfn_balance import TFNBalanceModel

SUPPORTED_TOY_ENCODERS = (
    "para_free",
    "hegnn",
    "hegnn_balance",
    "tfn",
    "tfn_balance",
    "mace",
    "mace_balance",
    "equiformer",
    "equiformer_balance",
)
ALLOWED_AUXILIARY_MISSING_PREFIXES = ("node_scalar_readout", "node_pos_proj")


def resolve_default_max_ell(model_type: str) -> int:
    del model_type
    return 6


def build_toy_encoder(
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
    grid_resolution: Optional[int] = None,
) -> nn.Module:
    model_type = str(model_type)
    if model_type not in SUPPORTED_TOY_ENCODERS:
        raise ValueError(f"Unsupported toy encoder: {model_type}")

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
        )

    common_kwargs: Dict[str, Any] = {
        "max_ell": int(max_ell),
        "num_layer": int(num_layers),
        "hidden_dim": int(hidden_dim),
        "irreps_channels": int(irreps_channels),
        "node_input_dim": 101,
        "edge_input_dim": 1,
    }

    if model_type == "hegnn":
        return HEGNNModel(**common_kwargs)
    if model_type == "hegnn_balance":
        return HEGNNBalanceModel(**common_kwargs, balance=True)
    if model_type == "tfn":
        return TFNModel(**common_kwargs)
    if model_type == "tfn_balance":
        return TFNBalanceModel(**common_kwargs, balance=True)
    if model_type == "mace":
        return MACEModel(**common_kwargs, correlation=int(mace_correlation))
    if model_type == "mace_balance":
        return MACEBalanceModel(
            **common_kwargs, correlation=int(mace_correlation), balance=True
        )
    if model_type == "equiformer":
        return EquiformerModel(**common_kwargs)
    if model_type == "equiformer_balance":
        return EquiformerBalanceModel(**common_kwargs)

    raise AssertionError(f"Unhandled toy encoder: {model_type}")


def load_checkpoint_payload(ckpt_path: str | Path) -> Dict[str, Any]:
    ckpt_path = Path(ckpt_path)
    if ckpt_path.suffix == ".npz":
        import json
        import numpy as np

        metadata_path = ckpt_path.with_suffix(".json")
        if not metadata_path.is_file():
            metadata_path = (
                Path(__file__).resolve().parents[2]
                / "configs/toy"
                / (ckpt_path.stem + ".json")
            )
        metadata = json.loads(metadata_path.read_text())
        with np.load(ckpt_path, allow_pickle=False) as arrays:
            state = {key: torch.from_numpy(arrays[key].copy()) for key in arrays.files}
        return {"state_dict": state, "hyper_parameters": metadata["hyper_parameters"]}
    payload = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported checkpoint payload type: {type(payload)!r}")
    return payload


def infer_checkpoint_family(payload: Dict[str, Any]) -> str:
    state_dict = payload.get("state_dict", {})
    hyper_parameters = payload.get("hyper_parameters", {})

    if any(str(key).startswith("classifier.") for key in state_dict):
        return "supervised"
    if "num_classes" in hyper_parameters:
        return "supervised"
    if "edge_loss_weight" in hyper_parameters:
        return "pretrained"
    return "encoder_only"


def extract_encoder_state_dict(payload: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    if "state_dict" in payload and isinstance(payload["state_dict"], dict):
        state_dict = payload["state_dict"]
    else:
        state_dict = payload

    backbone_state = {
        str(key)[len("backbone.") :]: value
        for key, value in state_dict.items()
        if str(key).startswith("backbone.")
    }
    if backbone_state:
        return backbone_state

    if any(str(key).startswith("classifier.") for key in state_dict):
        raise ValueError(
            "Checkpoint contains classifier weights but no backbone-prefixed encoder weights."
        )

    return {
        str(key): value for key, value in state_dict.items() if torch.is_tensor(value)
    }


def infer_model_type_from_checkpoint(
    ckpt_path: str | Path,
    payload: Optional[Dict[str, Any]] = None,
) -> str:
    resolved_payload = (
        payload if payload is not None else load_checkpoint_payload(ckpt_path)
    )
    hyper_parameters = dict(resolved_payload.get("hyper_parameters", {}))
    model_type = hyper_parameters.get("model_type")
    if model_type is None:
        raise ValueError(
            "Unable to infer model_type from checkpoint hyper_parameters. "
            "Pass --model explicitly for this checkpoint."
        )
    return str(model_type)


def load_encoder_from_checkpoint(
    ckpt_path: str | Path,
    *,
    model_type: Optional[str] = None,
    pf_component_mode: str = "none",
    pf_branch_mode: str = "hybrid",
    pf_body_local_scale: float = 1.0,
    pf_water_local_scale: float = 1.0,
    pf_relative_vector_scale: float = 1.0,
    grid_resolution: Optional[int] = None,
    device: str | torch.device = "cpu",
) -> Tuple[nn.Module, Dict[str, Any]]:
    payload = load_checkpoint_payload(ckpt_path)
    hyper_parameters = dict(payload.get("hyper_parameters", {}))
    checkpoint_model_type = infer_model_type_from_checkpoint(ckpt_path, payload=payload)
    resolved_model_type = (
        checkpoint_model_type if model_type is None else str(model_type)
    )

    if resolved_model_type != checkpoint_model_type:
        raise ValueError(
            f"Checkpoint model_type={checkpoint_model_type} does not match requested model_type={resolved_model_type}."
        )

    resolved_max_ell = int(
        hyper_parameters.get("max_ell", resolve_default_max_ell(resolved_model_type))
    )
    encoder = build_toy_encoder(
        model_type=resolved_model_type,
        hidden_dim=int(hyper_parameters.get("hidden_dim", 64)),
        num_layers=int(hyper_parameters.get("num_layers", 2)),
        max_ell=resolved_max_ell,
        irreps_channels=int(hyper_parameters.get("irreps_channels", 16)),
        mace_correlation=int(hyper_parameters.get("mace_correlation", 2)),
        pf_component_mode=pf_component_mode,
        pf_branch_mode=pf_branch_mode,
        pf_body_local_scale=pf_body_local_scale,
        pf_water_local_scale=pf_water_local_scale,
        pf_relative_vector_scale=pf_relative_vector_scale,
        grid_resolution=grid_resolution,
    )

    encoder_state = extract_encoder_state_dict(payload)
    missing_keys, unexpected_keys = encoder.load_state_dict(encoder_state, strict=False)
    allowed_missing_keys = [
        key
        for key in missing_keys
        if any(key.startswith(prefix) for prefix in ALLOWED_AUXILIARY_MISSING_PREFIXES)
    ]
    disallowed_missing_keys = [
        key for key in missing_keys if key not in allowed_missing_keys
    ]
    if disallowed_missing_keys or unexpected_keys:
        raise RuntimeError(
            "Failed to load encoder weights cleanly. "
            f"Missing keys: {missing_keys}. Unexpected keys: {unexpected_keys}."
        )

    encoder = encoder.to(device)
    encoder.eval()
    metadata = {
        "model_type": resolved_model_type,
        "init_mode": "checkpoint",
        "checkpoint_family": infer_checkpoint_family(payload),
        "checkpoint_path": str(Path(ckpt_path).resolve()),
        "hidden_dim": int(hyper_parameters.get("hidden_dim", 64)),
        "num_layers": int(hyper_parameters.get("num_layers", 2)),
        "max_ell": resolved_max_ell,
        "irreps_channels": int(hyper_parameters.get("irreps_channels", 16)),
        "mace_correlation": int(hyper_parameters.get("mace_correlation", 2)),
        "allowed_missing_keys": allowed_missing_keys,
        "pf_component_mode": str(pf_component_mode),
        "pf_branch_mode": str(pf_branch_mode),
        "pf_body_local_scale": float(pf_body_local_scale),
        "pf_water_local_scale": float(pf_water_local_scale),
        "pf_relative_vector_scale": float(pf_relative_vector_scale),
    }
    return encoder, metadata


def build_or_load_toy_encoder(
    *,
    model_type: Optional[str],
    ckpt_path: Optional[str],
    hidden_dim: int = 64,
    num_layers: int = 2,
    max_ell: Optional[int] = None,
    irreps_channels: int = 16,
    mace_correlation: int = 2,
    pf_component_mode: str = "none",
    pf_branch_mode: str = "hybrid",
    pf_body_local_scale: float = 1.0,
    pf_water_local_scale: float = 1.0,
    pf_relative_vector_scale: float = 1.0,
    grid_resolution: Optional[int] = None,
    device: str | torch.device = "cpu",
) -> Tuple[nn.Module, Dict[str, Any]]:
    if ckpt_path:
        return load_encoder_from_checkpoint(
            ckpt_path=ckpt_path,
            model_type=model_type,
            pf_component_mode=pf_component_mode,
            pf_branch_mode=pf_branch_mode,
            pf_body_local_scale=pf_body_local_scale,
            pf_water_local_scale=pf_water_local_scale,
            pf_relative_vector_scale=pf_relative_vector_scale,
            grid_resolution=grid_resolution,
            device=device,
        )

    if model_type is None:
        raise ValueError("model_type must be provided when ckpt_path is not set.")

    resolved_max_ell = (
        resolve_default_max_ell(model_type) if max_ell is None else int(max_ell)
    )
    encoder = build_toy_encoder(
        model_type=model_type,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        max_ell=resolved_max_ell,
        irreps_channels=irreps_channels,
        mace_correlation=mace_correlation,
        pf_component_mode=pf_component_mode,
        pf_branch_mode=pf_branch_mode,
        pf_body_local_scale=pf_body_local_scale,
        pf_water_local_scale=pf_water_local_scale,
        pf_relative_vector_scale=pf_relative_vector_scale,
        grid_resolution=grid_resolution,
    ).to(device)
    encoder.eval()

    init_mode = "parameter_free" if model_type == "para_free" else "random_init"
    metadata = {
        "model_type": str(model_type),
        "init_mode": init_mode,
        "checkpoint_family": None,
        "checkpoint_path": None,
        "hidden_dim": int(hidden_dim),
        "num_layers": int(num_layers),
        "max_ell": resolved_max_ell,
        "irreps_channels": int(irreps_channels),
        "mace_correlation": int(mace_correlation),
        "allowed_missing_keys": [],
        "pf_component_mode": str(pf_component_mode),
        "pf_branch_mode": str(pf_branch_mode),
        "pf_body_local_scale": float(pf_body_local_scale),
        "pf_water_local_scale": float(pf_water_local_scale),
        "pf_relative_vector_scale": float(pf_relative_vector_scale),
    }
    return encoder, metadata


__all__ = [
    "SUPPORTED_TOY_ENCODERS",
    "build_or_load_toy_encoder",
    "build_toy_encoder",
    "extract_encoder_state_dict",
    "infer_checkpoint_family",
    "infer_model_type_from_checkpoint",
    "load_encoder_from_checkpoint",
    "resolve_default_max_ell",
]
