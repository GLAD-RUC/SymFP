from typing import Optional
from functools import partial

import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool
from torch_scatter import scatter
from e3nn.o3 import Irreps, FullyConnectedTensorProduct, SphericalHarmonics

from .hegnn import BaseMLP, HEGNN_Layer, SH_INIT
from .utils import build_model_output


def _parity_complete_irreps(max_ell: int, irreps_channels: int) -> Irreps:
    parts = []
    for ell in range(max_ell + 1):
        parts.append(f"{irreps_channels}x{ell}e")
        parts.append(f"{irreps_channels}x{ell}o")
    return Irreps(" + ".join(parts)).sort()[0].simplify()


class BalancedSH_INIT(nn.Module):
    def __init__(
        self,
        hidden_dim,
        sh_irreps,
        hidden_irreps,
        activation=nn.SiLU(),
        pseudoscalar_eps: float = 1e-8,
    ):
        super().__init__()
        self.pseudoscalar_eps = pseudoscalar_eps
        self.spherical_harmonics = SphericalHarmonics(
            sh_irreps, normalize=True, normalization="norm"
        )
        self.sh_coff = FullyConnectedTensorProduct(
            sh_irreps, "1x0e + 1x0o", hidden_irreps, shared_weights=False
        )

        MLP = partial(BaseMLP, hidden_dim=hidden_dim, activation=activation)
        self.mlp_sh = MLP(input_dim=3 * hidden_dim + 1, output_dim=self.sh_coff.weight_numel)
        self.mlp_mix = MLP(input_dim=3 * hidden_dim + 1, output_dim=3)

    def forward(self, node_feat, node_pos, edge_index, edge_attr):
        row, col = edge_index
        diff_pos = node_pos[row] - node_pos[col]
        dist = torch.norm(diff_pos, dim=-1, keepdim=True)

        msg_input = torch.cat([dist, node_feat[row], node_feat[col], edge_attr], dim=-1)
        tp_weight = self.mlp_sh(msg_input)
        diff_sh = self.spherical_harmonics(diff_pos).detach()
        one = torch.ones([diff_sh.size(0), 1], device=diff_sh.device, dtype=diff_sh.dtype).detach()

        mix_weight = self.mlp_mix(msg_input)
        edge_vectors = mix_weight.unsqueeze(-1) * diff_pos.unsqueeze(1)
        node_vectors = scatter(
            edge_vectors,
            index=row,
            dim=0,
            dim_size=node_feat.size(0),
            reduce="mean",
        )
        v1, v2, v3 = node_vectors.unbind(dim=1)
        chi = torch.sum(v1 * torch.cross(v2, v3, dim=-1), dim=-1, keepdim=True)
        # Disabled for ablation: avoid amplifying near-zero pseudoscalars.
        # chi = chi / (chi.abs().detach() + self.pseudoscalar_eps)
        scalar_gate = torch.cat([one, chi[row]], dim=-1)

        diff_sh = self.sh_coff(diff_sh, scalar_gate, tp_weight)
        node_sh = scatter(diff_sh, index=row, dim=0, dim_size=node_feat.size(0), reduce="mean")
        return node_sh


class HEGNNBalanceModel(nn.Module):
    def __init__(
        self,
        max_ell: int = 11,
        num_layer: int = 4,
        hidden_dim: int = 64,
        irreps_channels: int = 16,
        node_input_dim: int = 11,
        edge_input_dim: int = 1,
        activation: nn.Module = nn.SiLU(),
        norm: Optional[nn.Module] = None,
        balance: bool = True,
        device: str = "cpu",
    ):
        super().__init__()

        sh_irreps = Irreps.spherical_harmonics(max_ell)
        self.spherical_harmonics = SphericalHarmonics(
            sh_irreps, normalize=True, normalization="norm"
        )

        self.irreps_channels = irreps_channels
        self.balance = balance
        self.hidden_irreps = (
            _parity_complete_irreps(max_ell, irreps_channels)
            if balance
            else (sh_irreps * irreps_channels).sort()[0].simplify()
        )
        self.output_dim = self.hidden_irreps.dim
        print(self.hidden_irreps)

        self.node_embedding = nn.Linear(node_input_dim, hidden_dim)
        self.edge_embedding = nn.Linear(edge_input_dim, hidden_dim)

        self.num_layer = num_layer
        self.sh_init = (
            BalancedSH_INIT(hidden_dim, sh_irreps, self.hidden_irreps, activation)
            if balance
            else SH_INIT(hidden_dim, sh_irreps, self.hidden_irreps, activation)
        )

        self.layers = nn.ModuleList()
        for _ in range(self.num_layer):
            layer = HEGNN_Layer(hidden_dim, self.hidden_irreps, activation=activation)
            self.layers.append(layer)

        self.to(device)

    def forward(self, data: Data):
        node_feat = self.node_embedding(data.x.float())
        node_pos = data.pos
        edge_index = data.edge_index
        edge_attr = self.edge_embedding(data.edge_attr)
        node_sh = self.sh_init(node_feat, node_pos, edge_index, edge_attr)

        for layer in self.layers:
            node_feat, node_pos, node_sh = layer(node_feat, node_pos, node_sh, edge_index, edge_attr)
        graph_feat = global_mean_pool(node_feat, data.batch)
        graph_sh = global_mean_pool(node_sh, data.batch)
        return build_model_output(
            node_feat=node_feat,
            node_sh=node_sh,
            graph_feat=graph_feat,
            graph_sh=graph_sh,
            node_pos=node_pos,
        )


HEGNNModel = HEGNNBalanceModel


__all__ = ["HEGNNBalanceModel", "HEGNNModel"]
