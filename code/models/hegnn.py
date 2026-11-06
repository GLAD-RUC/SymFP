from typing import Optional
from functools import partial

import torch
from torch import nn, LongTensor, Tensor
from torch.nn import functional as F
from torch_geometric.data import Data
from torch_geometric.nn import global_add_pool, global_mean_pool
from torch_scatter import scatter
import e3nn
from e3nn.o3 import Irreps, Linear, FullyConnectedTensorProduct, SphericalHarmonics

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

class HEGNN_Layer(nn.Module):
    def __init__(self, hidden_dim, hidden_irreps, activation=nn.SiLU()):
        super(HEGNN_Layer, self).__init__()
        self.hidden_irreps = hidden_irreps
        sh_channel = sum(mul for mul, ir in hidden_irreps)
        MLP = partial(BaseMLP, hidden_dim=hidden_dim, activation=activation)
        self.mlp_msg = MLP(input_dim=3 * hidden_dim + 1 + sh_channel, output_dim=hidden_dim, last_act=True)
        self.mlp_pos = MLP(input_dim=hidden_dim, output_dim=1)
        self.mlp_node_feat = MLP(input_dim=hidden_dim + hidden_dim, output_dim=hidden_dim)

        self.sh_msg = SH_Msg(hidden_irreps)
        self.sh_coff = FullyConnectedTensorProduct(
            self.hidden_irreps, '1x0e', self.hidden_irreps, shared_weights=False
        )
        self.mlp_sh = MLP(input_dim=hidden_dim, output_dim=self.sh_coff.weight_numel)

    def forward(self, node_feat, node_pos, node_sh, edge_index, edge_attr):
        msg, diff_pos, diff_sh = self.Msg(edge_index, edge_attr, node_feat, node_pos, node_sh)
        msg_agg, pos_agg, sh_agg = self.Agg(edge_index, node_feat.size(0), msg, diff_pos, diff_sh)
        node_feat, node_pos, node_sh = self.Upd(node_feat, node_pos, node_sh, msg_agg, pos_agg, sh_agg)
        return node_feat, node_pos, node_sh
        
    def Msg(self, edge_index, edge_attr, node_feat, node_pos, node_sh):
        row, col = edge_index
        diff_pos = node_pos[row] - node_pos[col]
        dist = torch.norm(diff_pos, p=2, dim=-1).unsqueeze(-1) ** 2
        sh_ip = self.sh_msg(edge_index, node_sh) 
        
        msg = torch.cat([node_feat[row], node_feat[col], edge_attr, dist, sh_ip], dim=-1)
        msg = self.mlp_msg(msg)
        diff_pos = diff_pos * self.mlp_pos(msg)

        diff_sh = node_sh[row] - node_sh[col]

        one = torch.ones([diff_sh.size(0), 1], device=diff_sh.device)
        diff_sh = self.sh_coff(diff_sh, one, self.mlp_sh(msg))

        return msg, diff_pos, diff_sh
    
    def Agg(self, edge_index, dim_size, msg, diff_pos, diff_sh):
        row, col = edge_index
        msg_agg = scatter(src=msg, index=row, dim=0, dim_size=dim_size, reduce='mean')
        pos_agg = scatter(src=diff_pos, index=row, dim=0, dim_size=dim_size, reduce='mean')
        sh_agg = scatter(src=diff_sh, index=row, dim=0, dim_size=dim_size, reduce='mean')
        return msg_agg, pos_agg, sh_agg
    
    def Upd(self, node_feat, node_pos, node_sh, msg_agg, pos_agg, sh_agg):
        node_sh = node_sh + sh_agg
        node_pos = node_pos + pos_agg
        node_feat = torch.cat([node_feat, msg_agg], dim=-1)
        node_feat = self.mlp_node_feat(node_feat)
        return node_feat, node_pos, node_sh
    
class SH_Msg(nn.Module):
    def __init__(self, hidden_irreps):
        super().__init__()
        self.hidden_irreps = hidden_irreps
        self.sh_channel = sum(mul for mul, ir in hidden_irreps)

    def forward(self, edge_index, node_sh):
        assert node_sh.size(1) == self.hidden_irreps.dim
        row, col = edge_index
        temp = node_sh[row] * node_sh[col]

        bs = node_sh[row].size(0)
        idx_res, idx_tmp = 0, 0
        ip = torch.zeros([bs, self.sh_channel], device=node_sh.device)
        for (mul, ir) in self.hidden_irreps:
            ip[:, idx_res:idx_res + mul] = torch.sum(temp[:, idx_tmp:idx_tmp + mul * ir.dim].view(bs, mul, ir.dim), dim=-1) / (mul ** 0.5)
            idx_res = idx_res + mul
            idx_tmp = idx_tmp + mul * ir.dim
        
        # Sometimes model will output NaN, then add the normalization for a better result. 
        # ip = ip / (torch.norm(ip, dim=1, keepdim=True).detach() + 1)
        return  ip

class SH_INIT(nn.Module):
    def __init__(self, hidden_dim, sh_irreps, hidden_irreps, activation=nn.SiLU()):
        super().__init__()
        self.spherical_harmonics = SphericalHarmonics(
            sh_irreps, normalize=True, normalization="norm"
        )

        self.sh_coff = FullyConnectedTensorProduct(
            sh_irreps, '1x0e', hidden_irreps, shared_weights=False
        )

        MLP = partial(BaseMLP, hidden_dim=hidden_dim, activation=activation)
        self.mlp_sh = MLP(input_dim=3 * hidden_dim + 1, output_dim=self.sh_coff.weight_numel)

    def forward(self, node_feat, node_pos, edge_index, edge_attr):
        row, col = edge_index
        diff_pos = node_pos[row] - node_pos[col]
        dist = torch.norm(diff_pos, dim=-1, keepdim=True)
        
        msg = torch.cat([dist, node_feat[row], node_feat[col], edge_attr], dim=-1)
        msg = self.mlp_sh(msg)
        diff_sh = self.spherical_harmonics(diff_pos).detach()
        one = torch.ones([diff_sh.size(0), 1], device=diff_sh.device).detach()
        diff_sh = self.sh_coff(diff_sh, one, msg)

        node_sh = scatter(diff_sh, index=row, dim=0, dim_size=node_feat.size(0), reduce='mean')

        return node_sh

class HEGNNModel(nn.Module):
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
        device: str = 'cpu',
    ):
        super().__init__()

        sh_irreps = Irreps.spherical_harmonics(max_ell)
        self.spherical_harmonics = SphericalHarmonics(
            sh_irreps, normalize=True, normalization="norm"
        )

        self.irreps_channels = irreps_channels
        self.hidden_irreps = (sh_irreps * irreps_channels).sort()[0].simplify()
        self.output_dim = self.hidden_irreps.dim
        print(self.hidden_irreps)

        self.node_embedding = nn.Linear(node_input_dim, hidden_dim)
        self.edge_embedding = nn.Linear(edge_input_dim, hidden_dim)
        
        self.num_layer = num_layer
        self.sh_init = SH_INIT(hidden_dim, sh_irreps, self.hidden_irreps, activation)

        self.layers = nn.ModuleList()
        for _ in range(self.num_layer):
            layer = HEGNN_Layer(hidden_dim, self.hidden_irreps, activation=activation)
            self.layers.append(layer)

        self.to(device)        

    @staticmethod
    def _resolve_batch(data: Data) -> Tensor:
        batch = getattr(data, "batch", None)
        if batch is None:
            return torch.zeros(data.pos.size(0), dtype=torch.long, device=data.pos.device)
        return batch

    @staticmethod
    def _resolve_pool_weights(data: Data) -> Optional[Tensor]:
        override = getattr(data, "node_weights_override", None)
        if override is None:
            return None
        if override.dim() == 1:
            return override.to(dtype=torch.float32).unsqueeze(-1)
        if override.dim() == 2 and override.size(-1) == 1:
            return override.to(dtype=torch.float32)
        raise ValueError(
            "node_weights_override must have shape [N] or [N, 1], "
            f"got {tuple(override.shape)}."
        )

    @staticmethod
    def _graph_pool(node_tensor: Tensor, batch: Tensor, weights: Optional[Tensor]) -> Tensor:
        if weights is None:
            return global_mean_pool(node_tensor, batch)
        num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
        weighted_sum = scatter(
            node_tensor * weights,
            batch,
            dim=0,
            dim_size=num_graphs,
            reduce="sum",
        )
        weight_sum = scatter(
            weights,
            batch,
            dim=0,
            dim_size=num_graphs,
            reduce="sum",
        ).clamp_min(1e-12)
        return weighted_sum / weight_sum

    def forward(self, data: Data):
        node_feat = self.node_embedding(data.x.float())
        node_pos = data.pos
        edge_index = data.edge_index
        edge_attr = self.edge_embedding(data.edge_attr)
        batch = self._resolve_batch(data)
        pool_weights = self._resolve_pool_weights(data)
        if pool_weights is not None:
            pool_weights = pool_weights.to(device=node_feat.device, dtype=node_feat.dtype)
        node_sh = self.sh_init(node_feat, node_pos, edge_index, edge_attr)

        for layer in self.layers:
            node_feat, node_pos, node_sh = layer(node_feat, node_pos, node_sh, edge_index, edge_attr)
        graph_feat = self._graph_pool(node_feat, batch, pool_weights)
        graph_sh = self._graph_pool(node_sh, batch, pool_weights)
        return build_model_output(
            node_feat=node_feat,
            node_sh=node_sh,
            graph_feat=graph_feat,
            graph_sh=graph_sh,
            node_pos=node_pos,
        )
