from typing import Callable, Optional, Tuple, Union

import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn import DimeNetPlusPlus
from torch_geometric.nn.models.dimenet import triplets
from torch_scatter import scatter

from .utils import build_model_output


def _atomic_numbers(x: Tensor) -> Tensor:
    if x.dim() == 1:
        return x.long()
    return x.argmax(dim=-1).long()


class DimeNetPPModel(DimeNetPlusPlus):
    def __init__(
        self,
        num_layer: int = 4,
        hidden_dim: int = 64,
        node_input_dim: int = 101,
        edge_input_dim: int = 1,
        int_emb_size: Optional[int] = None,
        basis_emb_size: int = 8,
        out_emb_channels: Optional[int] = None,
        num_spherical: int = 7,
        num_radial: int = 6,
        cutoff: float = 10.0,
        max_num_neighbors: int = 32,
        envelope_exponent: int = 5,
        num_before_skip: int = 1,
        num_after_skip: int = 2,
        num_output_layers: int = 3,
        act: Union[str, Callable] = "swish",
        device: str = "cpu",
        **kwargs,
    ) -> None:
        del node_input_dim, edge_input_dim, kwargs
        super().__init__(
            hidden_channels=hidden_dim,
            out_channels=hidden_dim,
            num_blocks=num_layer,
            int_emb_size=int_emb_size or hidden_dim,
            basis_emb_size=basis_emb_size,
            out_emb_channels=out_emb_channels or hidden_dim,
            num_spherical=num_spherical,
            num_radial=num_radial,
            cutoff=cutoff,
            max_num_neighbors=max_num_neighbors,
            envelope_exponent=envelope_exponent,
            num_before_skip=num_before_skip,
            num_after_skip=num_after_skip,
            num_output_layers=num_output_layers,
            act=act,
        )
        self.output_dim = hidden_dim
        self.to(device)

    def forward(self, data: Data):
        z = _atomic_numbers(data.x)

        i, j, idx_i, idx_j, idx_k, idx_kj, idx_ji = triplets(
            data.edge_index,
            num_nodes=z.size(0),
        )

        dist = (data.pos[i] - data.pos[j]).pow(2).sum(dim=-1).sqrt()

        pos_jk = data.pos[idx_j] - data.pos[idx_k]
        pos_ij = data.pos[idx_i] - data.pos[idx_j]
        a = (pos_ij * pos_jk).sum(dim=-1)
        b = torch.cross(pos_ij, pos_jk, dim=-1).norm(dim=-1)
        angle = torch.atan2(b, a)

        rbf = self.rbf(dist)
        sbf = self.sbf(dist, angle, idx_kj)

        x = self.emb(z, rbf, i, j)
        p = self.output_blocks[0](x, rbf, i, num_nodes=data.pos.size(0))

        for interaction_block, output_block in zip(self.interaction_blocks, self.output_blocks[1:]):
            x = interaction_block(x, rbf, sbf, idx_kj, idx_ji)
            p = p + output_block(x, rbf, i, num_nodes=data.pos.size(0))

        graph_feat = scatter(p, data.batch, dim=0, reduce="sum")
        return build_model_output(
            node_feat=p,
            node_sh=None,
            graph_feat=graph_feat,
            graph_sh=None,
        )
