from __future__ import annotations

from typing import Tuple

from models.dimenet import DimeNetPPModel
from models.egnn import EGNNModel
from models.equiformer import EquiformerModel
from models.equiformer_balance import EquiformerBalanceModel
from models.gvpgnn import GVPGNNModel
from models.hegnn import HEGNNModel
from models.hegnn_balance import HEGNNBalanceModel
from models.mace import MACEModel
from models.mace_balance import MACEBalanceModel
from models.schnet import SchNetModel
from models.tfn import TFNModel
from models.tfn_balance import TFNBalanceModel


def build_classification_backbone(
    *,
    model_type: str,
    hidden_dim: int,
    num_layers: int,
    max_ell: int,
    irreps_channels: int,
    mace_correlation: int,
) -> Tuple[object, int]:
    if model_type == "egnn":
        return (
            EGNNModel(
                num_layer=num_layers,
                hidden_dim=hidden_dim,
                node_input_dim=1,
                edge_input_dim=1,
            ),
            hidden_dim,
        )
    if model_type == "equiformer":
        return (
            EquiformerModel(
                max_ell=max_ell,
                num_layer=num_layers,
                hidden_dim=hidden_dim,
                irreps_channels=irreps_channels,
                node_input_dim=101,
                edge_input_dim=1,
            ),
            hidden_dim,
        )
    if model_type == "equiformer_balance":
        return (
            EquiformerBalanceModel(
                max_ell=max_ell,
                num_layer=num_layers,
                hidden_dim=hidden_dim,
                irreps_channels=irreps_channels,
                node_input_dim=101,
                edge_input_dim=1,
            ),
            hidden_dim,
        )
    if model_type == "hegnn":
        return (
            HEGNNModel(
                max_ell=max_ell,
                num_layer=num_layers,
                hidden_dim=hidden_dim,
                irreps_channels=irreps_channels,
                node_input_dim=101,
                edge_input_dim=1,
            ),
            hidden_dim,
        )
    if model_type == "hegnn_balance":
        return (
            HEGNNBalanceModel(
                max_ell=max_ell,
                num_layer=num_layers,
                hidden_dim=hidden_dim,
                irreps_channels=irreps_channels,
                node_input_dim=101,
                edge_input_dim=1,
                balance=True,
            ),
            hidden_dim,
        )
    if model_type == "schnet":
        return (
            SchNetModel(
                num_layer=num_layers,
                hidden_dim=hidden_dim,
                node_input_dim=101,
                edge_input_dim=1,
            ),
            hidden_dim,
        )
    if model_type == "dimenet":
        return (
            DimeNetPPModel(
                num_layer=num_layers,
                hidden_dim=hidden_dim,
                node_input_dim=101,
                edge_input_dim=1,
            ),
            hidden_dim,
        )
    if model_type == "gvp":
        return (
            GVPGNNModel(
                num_layer=num_layers,
                hidden_dim=hidden_dim,
                irreps_channels=irreps_channels,
                node_input_dim=101,
                edge_input_dim=1,
            ),
            hidden_dim,
        )
    if model_type == "tfn":
        return (
            TFNModel(
                max_ell=max_ell,
                num_layer=num_layers,
                hidden_dim=hidden_dim,
                irreps_channels=irreps_channels,
                node_input_dim=101,
                edge_input_dim=1,
            ),
            hidden_dim,
        )
    if model_type == "tfn_balance":
        return (
            TFNBalanceModel(
                max_ell=max_ell,
                num_layer=num_layers,
                hidden_dim=hidden_dim,
                irreps_channels=irreps_channels,
                node_input_dim=101,
                edge_input_dim=1,
                balance=True,
            ),
            hidden_dim,
        )
    if model_type == "mace":
        return (
            MACEModel(
                max_ell=max_ell,
                correlation=mace_correlation,
                num_layer=num_layers,
                hidden_dim=hidden_dim,
                irreps_channels=irreps_channels,
                node_input_dim=101,
                edge_input_dim=1,
            ),
            hidden_dim,
        )
    if model_type == "mace_balance":
        return (
            MACEBalanceModel(
                max_ell=max_ell,
                correlation=mace_correlation,
                num_layer=num_layers,
                hidden_dim=hidden_dim,
                irreps_channels=irreps_channels,
                node_input_dim=101,
                edge_input_dim=1,
                balance=True,
            ),
            hidden_dim,
        )
    raise ValueError(f"Unknown model type: {model_type}")


def build_denoise_backbone(
    *,
    model_type: str,
    hidden_dim: int,
    num_layers: int,
    max_ell: int,
    irreps_channels: int,
    mace_correlation: int,
):
    if model_type == "equiformer":
        return EquiformerModel(
            max_ell=max_ell,
            num_layer=num_layers,
            hidden_dim=hidden_dim,
            irreps_channels=irreps_channels,
            node_input_dim=101,
            edge_input_dim=1,
        )
    if model_type == "equiformer_balance":
        return EquiformerBalanceModel(
            max_ell=max_ell,
            num_layer=num_layers,
            hidden_dim=hidden_dim,
            irreps_channels=irreps_channels,
            node_input_dim=101,
            edge_input_dim=1,
        )
    if model_type == "hegnn":
        return HEGNNModel(
            max_ell=max_ell,
            num_layer=num_layers,
            hidden_dim=hidden_dim,
            irreps_channels=irreps_channels,
            node_input_dim=101,
            edge_input_dim=1,
        )
    if model_type == "hegnn_balance":
        return HEGNNBalanceModel(
            max_ell=max_ell,
            num_layer=num_layers,
            hidden_dim=hidden_dim,
            irreps_channels=irreps_channels,
            node_input_dim=101,
            edge_input_dim=1,
            balance=True,
        )
    if model_type == "tfn":
        return TFNModel(
            max_ell=max_ell,
            num_layer=num_layers,
            hidden_dim=hidden_dim,
            irreps_channels=irreps_channels,
            node_input_dim=101,
            edge_input_dim=1,
        )
    if model_type == "tfn_balance":
        return TFNBalanceModel(
            max_ell=max_ell,
            num_layer=num_layers,
            hidden_dim=hidden_dim,
            irreps_channels=irreps_channels,
            node_input_dim=101,
            edge_input_dim=1,
            balance=True,
        )
    if model_type == "mace":
        return MACEModel(
            max_ell=max_ell,
            correlation=mace_correlation,
            num_layer=num_layers,
            hidden_dim=hidden_dim,
            irreps_channels=irreps_channels,
            node_input_dim=101,
            edge_input_dim=1,
        )
    if model_type == "mace_balance":
        return MACEBalanceModel(
            max_ell=max_ell,
            correlation=mace_correlation,
            num_layer=num_layers,
            hidden_dim=hidden_dim,
            irreps_channels=irreps_channels,
            node_input_dim=101,
            edge_input_dim=1,
            balance=True,
        )
    raise ValueError(f"Unsupported model type: {model_type}")


__all__ = [
    "build_classification_backbone",
    "build_denoise_backbone",
]
