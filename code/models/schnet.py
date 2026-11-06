from typing import Optional, Tuple

import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn import SchNet, global_add_pool, global_mean_pool

from .utils import build_model_output


def _atomic_numbers(x: Tensor) -> Tensor:
    if x.dim() == 1:
        return x.long()
    return x.argmax(dim=-1).long()


class SchNetModel(SchNet):
    def __init__(
        self,
        num_layer: int = 4,
        hidden_dim: int = 64,
        node_input_dim: int = 101,
        edge_input_dim: int = 1,
        num_filters: Optional[int] = None,
        num_gaussians: int = 50,
        cutoff: float = 10.0,
        max_num_neighbors: int = 32,
        pool: str = "mean",
        device: str = "cpu",
        **kwargs,
    ) -> None:
        del node_input_dim, edge_input_dim, kwargs
        super().__init__(
            hidden_channels=hidden_dim,
            num_filters=num_filters or hidden_dim,
            num_interactions=num_layer,
            num_gaussians=num_gaussians,
            cutoff=cutoff,
            interaction_graph=None,
            max_num_neighbors=max_num_neighbors,
            readout=pool,
            dipole=False,
            mean=None,
            std=None,
            atomref=None,
        )
        self.pool = {"mean": global_mean_pool, "sum": global_add_pool}[pool]
        self.output_dim = hidden_dim
        self.to(device)

    def forward(self, data: Data):
        z = _atomic_numbers(data.x)
        h = self.embedding(z)

        row, col = data.edge_index
        edge_weight = (data.pos[row] - data.pos[col]).norm(dim=-1)
        edge_attr = self.distance_expansion(edge_weight)

        for interaction in self.interactions:
            h = h + interaction(h, data.edge_index, edge_weight, edge_attr)

        graph_feat = self.pool(h, data.batch)
        return build_model_output(
            node_feat=h,
            node_sh=None,
            graph_feat=graph_feat,
            graph_sh=None,
        )
