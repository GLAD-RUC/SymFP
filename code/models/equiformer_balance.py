from __future__ import annotations

from typing import Optional

import torch
from e3nn import o3
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import global_add_pool, global_mean_pool

from .equiformer_modules import (
    EdgeDegreeEmbeddingNetwork,
    GaussianRadialBasisLayer,
    LinearRS,
    TransBlock,
    _non_scalar_irreps,
    _parity_complete_irreps,
    _scale_irreps_for_heads,
    get_norm_layer,
)
from .utils import build_model_output


class EquiformerBalanceModel(nn.Module):
    def __init__(
        self,
        max_ell: int = 2,
        num_layer: int = 4,
        hidden_dim: int = 64,
        irreps_channels: int = 16,
        node_input_dim: int = 101,
        edge_input_dim: int = 1,
        number_of_basis: int = 32,
        radial_hidden_dim: int = 64,
        num_heads: int = 4,
        hidden_irreps: Optional[o3.Irreps] = None,
        head_irreps: Optional[o3.Irreps] = None,
        mlp_mid_irreps: Optional[o3.Irreps] = None,
        r_max: float = 10.0,
        pool: str = "mean",
        avg_degree: float = 16.0,
        rescale_degree: bool = False,
        nonlinear_message: bool = False,
        norm_layer: Optional[str] = "layer",
        alpha_drop: float = 0.0,
        proj_drop: float = 0.0,
        drop_path_rate: float = 0.0,
        device: str = "cpu",
        **kwargs,
    ):
        del edge_input_dim, kwargs
        super().__init__()
        self.max_ell = max_ell
        self.hidden_dim = hidden_dim
        self.irreps_channels = irreps_channels
        self.num_heads = num_heads
        self.output_dim = hidden_dim

        self.sh_irreps = o3.Irreps.spherical_harmonics(self.max_ell)
        if hidden_irreps is None:
            hidden_irreps = _parity_complete_irreps(self.max_ell, self.irreps_channels)
        self.hidden_irreps = o3.Irreps(hidden_irreps)
        self.head_irreps = (
            o3.Irreps(head_irreps)
            if head_irreps is not None
            else _scale_irreps_for_heads(self.hidden_irreps, self.num_heads)
        )
        self.mlp_mid_irreps = (
            o3.Irreps(mlp_mid_irreps)
            if mlp_mid_irreps is not None
            else self.hidden_irreps
        )
        self.non_scalar_irreps = _non_scalar_irreps(self.hidden_irreps)

        scalar_mul = sum(mul for mul, ir in self.hidden_irreps if ir.l == 0 and ir.p == 1)
        vector_mul = sum(mul for mul, ir in self.non_scalar_irreps if ir.l == 1 and ir.p == -1)
        if scalar_mul == 0:
            raise ValueError("EquiformerBalance hidden_irreps must contain at least one 0e channel.")
        if vector_mul == 0:
            raise ValueError("EquiformerBalance hidden_irreps must contain at least one 1o channel.")

        self.scalar_embedding = nn.Linear(node_input_dim, hidden_dim)
        self.node_embedding = LinearRS(o3.Irreps(f"{hidden_dim}x0e"), self.hidden_irreps)
        self.rbf = GaussianRadialBasisLayer(number_of_basis, cutoff=r_max)
        self.edge_degree_embedding = EdgeDegreeEmbeddingNetwork(
            irreps_node_embedding=self.hidden_irreps,
            irreps_edge_attr=self.sh_irreps,
            fc_neurons=[number_of_basis, radial_hidden_dim, radial_hidden_dim],
            avg_aggregate_num=avg_degree,
        )
        self.blocks = nn.ModuleList(
            [
                TransBlock(
                    irreps_node_input=self.hidden_irreps,
                    irreps_node_attr=o3.Irreps("1x0e"),
                    irreps_edge_attr=self.sh_irreps,
                    irreps_node_output=self.hidden_irreps,
                    fc_neurons=[number_of_basis, radial_hidden_dim, radial_hidden_dim],
                    irreps_head=self.head_irreps,
                    num_heads=self.num_heads,
                    irreps_pre_attn=None,
                    rescale_degree=rescale_degree,
                    nonlinear_message=nonlinear_message,
                    alpha_drop=alpha_drop,
                    proj_drop=proj_drop,
                    drop_path_rate=drop_path_rate,
                    irreps_mlp_mid=self.mlp_mid_irreps,
                    norm_layer=norm_layer,
                )
                for _ in range(num_layer)
            ]
        )
        norm = get_norm_layer(norm_layer)
        self.norm = norm(self.hidden_irreps) if norm is not None else None
        self.node_scalar_readout = LinearRS(self.hidden_irreps, o3.Irreps(f"{hidden_dim}x0e"))
        self.node_pos_proj = o3.Linear(self.hidden_irreps, o3.Irreps("1x1o"))
        self.scalar_readout = LinearRS(self.hidden_irreps, o3.Irreps(f"{hidden_dim}x0e"))
        self.pool = {"mean": global_mean_pool, "sum": global_add_pool}[pool]
        self.to(device)

    def forward(self, data: Data):
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(data.pos.size(0), dtype=torch.long, device=data.pos.device)

        node_state = self.node_embedding(self.scalar_embedding(data.x.float()))
        edge_src, edge_dst = data.edge_index
        edge_vec = data.pos[edge_src] - data.pos[edge_dst]
        edge_sh = o3.spherical_harmonics(
            self.sh_irreps,
            edge_vec,
            normalize=True,
            normalization="component",
        )
        edge_rbf = self.rbf(edge_vec.norm(dim=-1))
        node_state = node_state + self.edge_degree_embedding(
            node_input=node_state,
            edge_attr=edge_sh,
            edge_scalars=edge_rbf,
            edge_src=edge_src,
            edge_dst=edge_dst,
            batch=batch,
        )
        node_attr = torch.ones_like(node_state[:, :1])

        for block in self.blocks:
            node_state = block(
                node_input=node_state,
                node_attr=node_attr,
                edge_src=edge_src,
                edge_dst=edge_dst,
                edge_attr=edge_sh,
                edge_scalars=edge_rbf,
                batch=batch,
            )

        if self.norm is not None:
            node_state = self.norm(node_state, batch=batch)

        node_feat = self.node_scalar_readout(node_state)
        node_sh = node_state
        node_pos = data.pos + self.node_pos_proj(node_state)

        graph_irreps = self.pool(node_state, batch)
        graph_feat = self.scalar_readout(graph_irreps)
        graph_sh = graph_irreps

        return build_model_output(
            node_feat=node_feat,
            node_sh=node_sh,
            graph_feat=graph_feat,
            graph_sh=graph_sh,
            node_pos=node_pos,
            node_irreps=node_state,
            graph_irreps=graph_irreps,
        )


__all__ = ["EquiformerBalanceModel"]
