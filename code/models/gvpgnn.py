import functools
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import torch_scatter
from torch import Tensor, nn
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing, global_add_pool, global_mean_pool

from .mace_modules.blocks import RadialEmbeddingBlock
from .utils import build_model_output


def tuple_sum(*args):
    return tuple(map(sum, zip(*args)))


def tuple_cat(*args, dim: int = -1):
    dim %= len(args[0][0].shape)
    s_args, v_args = list(zip(*args))
    return torch.cat(s_args, dim=dim), torch.cat(v_args, dim=dim)


def tuple_index(x, idx):
    return x[0][idx], x[1][idx]


def _norm_no_nan(x, axis: int = -1, keepdims: bool = False, eps: float = 1e-8, sqrt: bool = True):
    out = torch.clamp(torch.sum(torch.square(x), axis, keepdims), min=eps)
    return torch.sqrt(out) if sqrt else out


def _split(x: Tensor, nv: int):
    s = x[..., :-3 * nv]
    v = x[..., -3 * nv:].contiguous().view(x.shape[0], nv, 3)
    return s, v


def _merge(s: Tensor, v: Tensor) -> Tensor:
    v = v.contiguous().view(v.shape[0], v.shape[1] * 3)
    return torch.cat([s, v], -1)


class GVP(nn.Module):
    def __init__(
        self,
        in_dims,
        out_dims,
        h_dim: Optional[int] = None,
        activations=(F.relu, torch.sigmoid),
        vector_gate: bool = True,
    ):
        super().__init__()
        self.si, self.vi = in_dims
        self.so, self.vo = out_dims
        self.vector_gate = vector_gate
        if self.vi:
            self.h_dim = h_dim or max(self.vi, self.vo)
            self.wh = nn.Linear(self.vi, self.h_dim, bias=False)
            self.ws = nn.Linear(self.h_dim + self.si, self.so)
            if self.vo:
                self.wv = nn.Linear(self.h_dim, self.vo, bias=False)
                if self.vector_gate:
                    self.wsv = nn.Linear(self.so, self.vo)
        else:
            self.ws = nn.Linear(self.si, self.so)

        self.scalar_act, self.vector_act = activations
        self.dummy_param = nn.Parameter(torch.empty(0))

    def forward(self, x):
        if self.vi:
            s, v = x
            v = torch.transpose(v, -1, -2)
            vh = self.wh(v)
            vn = _norm_no_nan(vh, axis=-2)
            s = self.ws(torch.cat([s, vn], -1))
            if self.vo:
                v = self.wv(vh)
                v = torch.transpose(v, -1, -2)
                if self.vector_gate:
                    gate = self.wsv(self.vector_act(s)) if self.vector_act else self.wsv(s)
                    v = v * torch.sigmoid(gate).unsqueeze(-1)
                elif self.vector_act:
                    v = v * self.vector_act(_norm_no_nan(v, axis=-1, keepdims=True))
        else:
            s = self.ws(x)
            if self.vo:
                v = torch.zeros(s.shape[0], self.vo, 3, device=self.dummy_param.device)
        if self.scalar_act:
            s = self.scalar_act(s)
        return (s, v) if self.vo else s


class _VDropout(nn.Module):
    def __init__(self, drop_rate: float):
        super().__init__()
        self.drop_rate = drop_rate
        self.dummy_param = nn.Parameter(torch.empty(0))

    def forward(self, x: Tensor) -> Tensor:
        if not self.training:
            return x
        mask = torch.bernoulli(
            (1 - self.drop_rate) * torch.ones(x.shape[:-1], device=self.dummy_param.device)
        ).unsqueeze(-1)
        return mask * x / (1 - self.drop_rate)


class Dropout(nn.Module):
    def __init__(self, drop_rate: float):
        super().__init__()
        self.sdropout = nn.Dropout(drop_rate)
        self.vdropout = _VDropout(drop_rate)

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            return self.sdropout(x)
        s, v = x
        return self.sdropout(s), self.vdropout(v)


class LayerNorm(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.s, self.v = dims
        self.scalar_norm = nn.LayerNorm(self.s)

    def forward(self, x):
        if not self.v:
            return self.scalar_norm(x)
        s, v = x
        vn = _norm_no_nan(v, axis=-1, keepdims=True, sqrt=False)
        vn = torch.sqrt(torch.mean(vn, dim=-2, keepdim=True))
        return self.scalar_norm(s), v / vn


class GVPConv(MessagePassing):
    def __init__(
        self,
        in_dims,
        out_dims,
        edge_dims,
        n_layers: int = 3,
        module_list=None,
        aggr: str = "mean",
        activations=(F.relu, torch.sigmoid),
        vector_gate: bool = True,
    ):
        super().__init__(aggr=aggr)
        self.si, self.vi = in_dims
        self.so, self.vo = out_dims
        self.se, self.ve = edge_dims

        gvp_layer = functools.partial(GVP, activations=activations, vector_gate=vector_gate)
        module_list = module_list or []
        if not module_list:
            if n_layers == 1:
                module_list.append(
                    gvp_layer(
                        (2 * self.si + self.se, 2 * self.vi + self.ve),
                        (self.so, self.vo),
                        activations=(None, None),
                    )
                )
            else:
                module_list.append(gvp_layer((2 * self.si + self.se, 2 * self.vi + self.ve), out_dims))
                for _ in range(n_layers - 2):
                    module_list.append(gvp_layer(out_dims, out_dims))
                module_list.append(gvp_layer(out_dims, out_dims, activations=(None, None)))
        self.message_func = nn.Sequential(*module_list)

    def forward(self, x, edge_index: Tensor, edge_attr):
        x_s, x_v = x
        message = self.propagate(
            edge_index,
            s=x_s,
            v=x_v.contiguous().view(x_v.shape[0], x_v.shape[1] * 3),
            edge_attr=edge_attr,
        )
        return _split(message, self.vo)

    def message(self, s_i, v_i, s_j, v_j, edge_attr):
        v_j = v_j.view(v_j.shape[0], v_j.shape[1] // 3, 3)
        v_i = v_i.view(v_i.shape[0], v_i.shape[1] // 3, 3)
        message = tuple_cat((s_j, v_j), edge_attr, (s_i, v_i))
        message = self.message_func(message)
        return _merge(*message)


class GVPConvLayer(nn.Module):
    def __init__(
        self,
        node_dims,
        edge_dims,
        n_message: int = 3,
        n_feedforward: int = 2,
        drop_rate: float = 0.1,
        autoregressive: bool = False,
        activations=(F.relu, torch.sigmoid),
        vector_gate: bool = True,
        residual: bool = True,
    ):
        super().__init__()
        self.conv = GVPConv(
            node_dims,
            node_dims,
            edge_dims,
            n_message,
            aggr="add" if autoregressive else "mean",
            activations=activations,
            vector_gate=vector_gate,
        )
        gvp_layer = functools.partial(GVP, activations=activations, vector_gate=vector_gate)
        self.norm = nn.ModuleList([LayerNorm(node_dims) for _ in range(2)])
        self.dropout = nn.ModuleList([Dropout(drop_rate) for _ in range(2)])

        ff_func = []
        if n_feedforward == 1:
            ff_func.append(gvp_layer(node_dims, node_dims, activations=(None, None)))
        else:
            hid_dims = 4 * node_dims[0], 2 * node_dims[1]
            ff_func.append(gvp_layer(node_dims, hid_dims))
            ff_func.extend(gvp_layer(hid_dims, hid_dims) for _ in range(n_feedforward - 2))
            ff_func.append(gvp_layer(hid_dims, node_dims, activations=(None, None)))
        self.ff_func = nn.Sequential(*ff_func)
        self.residual = residual

    def forward(self, x, edge_index: Tensor, edge_attr, autoregressive_x=None, node_mask=None):
        if autoregressive_x is not None:
            src, dst = edge_index
            mask = src < dst
            edge_index_forward = edge_index[:, mask]
            edge_index_backward = edge_index[:, ~mask]
            edge_attr_forward = tuple_index(edge_attr, mask)
            edge_attr_backward = tuple_index(edge_attr, ~mask)
            dh = tuple_sum(
                self.conv(x, edge_index_forward, edge_attr_forward),
                self.conv(autoregressive_x, edge_index_backward, edge_attr_backward),
            )
            count = (
                torch_scatter.scatter_add(torch.ones_like(dst), dst, dim_size=dh[0].size(0))
                .clamp(min=1)
                .unsqueeze(-1)
            )
            dh = dh[0] / count, dh[1] / count.unsqueeze(-1)
        else:
            dh = self.conv(x, edge_index, edge_attr)

        if node_mask is not None:
            x_ = x
            x, dh = tuple_index(x, node_mask), tuple_index(dh, node_mask)

        x = self.norm[0](tuple_sum(x, self.dropout[0](dh))) if self.residual else dh
        dh = self.ff_func(x)
        x = self.norm[1](tuple_sum(x, self.dropout[1](dh))) if self.residual else dh

        if node_mask is not None:
            x_[0][node_mask], x_[1][node_mask] = x[0], x[1]
            x = x_
        return x


class GVPGNNModel(torch.nn.Module):
    def __init__(
        self,
        num_layer: int = 4,
        hidden_dim: int = 64,
        irreps_channels: int = 16,
        node_input_dim: int = 101,
        edge_input_dim: int = 1,
        r_max: float = 10.0,
        num_bessel: int = 8,
        num_polynomial_cutoff: int = 5,
        s_dim_edge: int = 32,
        v_dim_edge: int = 1,
        pool: str = "mean",
        residual: bool = True,
        device: str = "cpu",
        **kwargs,
    ) -> None:
        del edge_input_dim, kwargs
        super().__init__()

        self.hidden_dim = hidden_dim
        self.output_dim = hidden_dim

        activations = (F.relu, None)
        node_dims = (hidden_dim, irreps_channels)
        edge_dims = (s_dim_edge, v_dim_edge)

        self.node_embedding = nn.Linear(node_input_dim, hidden_dim)
        self.W_v = nn.Sequential(
            LayerNorm((hidden_dim, 0)),
            GVP((hidden_dim, 0), node_dims, activations=(None, None), vector_gate=True),
        )

        self.radial_embedding = RadialEmbeddingBlock(
            r_max=r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
        )
        self.W_e = nn.Sequential(
            LayerNorm((self.radial_embedding.out_dim, 1)),
            GVP(
                (self.radial_embedding.out_dim, 1),
                edge_dims,
                activations=(None, None),
                vector_gate=True,
            ),
        )

        self.layers = nn.ModuleList(
            GVPConvLayer(
                node_dims,
                edge_dims,
                activations=activations,
                vector_gate=True,
                residual=residual,
            )
            for _ in range(num_layer)
        )

        self.pool = {"mean": global_mean_pool, "sum": global_add_pool}[pool]
        self.to(device)

    def forward(self, data: Data):
        vectors = data.pos[data.edge_index[0]] - data.pos[data.edge_index[1]]
        lengths = torch.linalg.norm(vectors, dim=-1, keepdim=True)

        h_v = self.node_embedding(data.x.float())
        h_e = (
            self.radial_embedding(lengths),
            torch.nan_to_num(torch.div(vectors, lengths)).unsqueeze(-2),
        )

        h_v = self.W_v(h_v)
        h_e = self.W_e(h_e)

        for layer in self.layers:
            h_v = layer(h_v, data.edge_index, h_e)

        node_feat = h_v[0]
        node_sh = h_v[1].reshape(h_v[1].shape[0], -1)
        graph_feat = self.pool(node_feat, data.batch)
        return build_model_output(
            node_feat=node_feat,
            node_sh=node_sh,
            graph_feat=graph_feat,
            graph_sh=None,
        )
