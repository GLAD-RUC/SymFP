from typing import Optional, Tuple, Union, Dict, List
from functools import partial

import torch
from torch import nn, LongTensor, Tensor
from torch.nn import functional as F
from torch_geometric.data import Data
from torch_geometric.nn import global_add_pool, global_mean_pool
from torch_scatter import scatter

from .utils import build_model_output


class BaseMLP(nn.Module):
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int, 
        output_dim: int, 
        activation: nn.Module = nn.SiLU(),
        norm: Optional[nn.Module] = None,
        residual: bool = False, 
        last_act: bool = False,
    ) -> None:
        super(BaseMLP, self).__init__()
        self.residual = residual
        if residual:
            assert output_dim == input_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Identity() if norm is None else norm(hidden_dim),
            activation,
            nn.Linear(hidden_dim, output_dim),
            nn.Identity() if norm is None else norm(output_dim),
            activation if last_act else nn.Identity()
        )

    def forward(self, x):
        return x + self.mlp(x) if self.residual else self.mlp(x)

class EGNNLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 64,
        edge_attr_dim: int = 2,
        activation: nn.Module = nn.SiLU(),
        norm: Optional[nn.Module] = None,
    ) -> None:
        super(EGNNLayer, self).__init__()
        MLP = partial(BaseMLP, hidden_dim=hidden_dim, activation=activation, norm=norm)
        self.mlp_msg = MLP(input_dim=2 * hidden_dim + edge_attr_dim + 1, output_dim=hidden_dim, last_act=True)
        self.mlp_pos = MLP(input_dim=hidden_dim, output_dim=1)
        self.mlp_node_feat = MLP(input_dim=hidden_dim + hidden_dim, output_dim=hidden_dim)

    def forward(
        self,
        node_feat: Tensor, 
        node_pos: Tensor, 
        edge_index: List[LongTensor], 
        edge_attr: Optional[Tensor],
    ) -> Tensor:
        msg, diff_pos = self.Msg(edge_index, edge_attr, node_feat, node_pos)
        msg_agg, pos_agg = self.Agg(edge_index, node_feat.size(0), msg, diff_pos)
        node_feat, node_pos = self.Upd(node_feat, node_pos, msg_agg, pos_agg)
        return node_feat, node_pos

    def Msg(self, edge_index, edge_attr, node_feat, node_pos):
        row, col = edge_index
        diff_pos = node_pos[row] - node_pos[col]
        dist = torch.norm(diff_pos, p=2, dim=-1).unsqueeze(-1) ** 2
        
        msg = torch.cat([i for i in [node_feat[row], node_feat[col], edge_attr, dist] if i is not None], dim=-1)
        msg = self.mlp_msg(msg)
        diff_pos = diff_pos * self.mlp_pos(msg)

        return msg, diff_pos
    
    def Agg(self, edge_index, dim_size, msg, diff_pos):
        row, col = edge_index
        msg_agg = scatter(src=msg, index=row, dim=0, dim_size=dim_size, reduce='mean')
        pos_agg = scatter(src=diff_pos, index=row, dim=0, dim_size=dim_size, reduce='mean')
        return msg_agg, pos_agg
    
    def Upd(self, node_feat, node_pos, msg_agg, pos_agg):
        node_pos = node_pos + pos_agg

        node_feat = torch.cat([node_feat, msg_agg], dim=-1)
        node_feat = self.mlp_node_feat(node_feat)
        return node_feat, node_pos

class EGNNModel(torch.nn.Module):
    def __init__(
        self,
        num_layer: int = 4,
        hidden_dim: int = 64,
        node_input_dim: int = 1,
        edge_input_dim: int = 1,
        activation: nn.Module = nn.SiLU(),
        norm: Optional[nn.Module] = None,
        device: str = 'cpu',
        **kwargs,
    ) -> None:
        super(EGNNModel, self).__init__()
        self.node_embedding = nn.Linear(node_input_dim, hidden_dim)
        self.edge_embedding = nn.Linear(edge_input_dim, hidden_dim)

        self.layers = torch.nn.ModuleList()
        for _ in range(num_layer):
            self.layers.append(EGNNLayer(hidden_dim, hidden_dim, activation, norm))

        self.to(device)

    def forward(self, data: Data):
        node_feat = self.node_embedding(data.x.float())
        node_pos = data.pos
        edge_index = data.edge_index
        edge_attr = self.edge_embedding(data.edge_attr)

        for layer in self.layers:
            node_feat, node_pos = layer(node_feat, node_pos, edge_index, edge_attr)
        graph_feat = global_mean_pool(node_feat, data.batch)
        return build_model_output(
            node_feat=node_feat,
            node_sh=None,
            graph_feat=graph_feat,
            graph_sh=None,
            node_pos=node_pos,
        )
