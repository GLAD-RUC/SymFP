from typing import Optional, Tuple

import e3nn
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch_geometric.data import Data
from torch_geometric.nn import global_add_pool, global_mean_pool
from torch_scatter import scatter

from .mace_modules.blocks import RadialEmbeddingBlock
from .mace_modules.irreps_tools import irreps2gate
from .utils import build_model_output


def _non_scalar_irreps(irreps: e3nn.o3.Irreps) -> e3nn.o3.Irreps:
    parts = [(mul, ir) for mul, ir in irreps if not (ir.l == 0 and ir.p == 1)]
    return e3nn.o3.Irreps(parts)


class TensorProductConvLayer(torch.nn.Module):
    def __init__(
        self,
        in_irreps,
        out_irreps,
        sh_irreps,
        edge_feats_dim: int,
        mlp_dim: int,
        aggr: str = "add",
        batch_norm: bool = False,
        gate: bool = False,
    ):
        super().__init__()
        self.in_irreps = in_irreps
        self.out_irreps = out_irreps
        self.sh_irreps = sh_irreps
        self.edge_feats_dim = edge_feats_dim
        self.aggr = aggr

        if gate:
            irreps_scalars, irreps_gates, irreps_gated = irreps2gate(e3nn.o3.Irreps(out_irreps))
            act_scalars = [torch.nn.functional.silu for _ in irreps_scalars]
            act_gates = [torch.sigmoid for _ in irreps_gates]
            if irreps_gated.num_irreps == 0:
                self.gate = e3nn.nn.Activation(out_irreps, acts=[torch.nn.functional.silu])
            else:
                self.gate = e3nn.nn.Gate(
                    irreps_scalars,
                    act_scalars,
                    irreps_gates,
                    act_gates,
                    irreps_gated,
                )
                out_irreps = self.gate.irreps_in
        else:
            self.gate = None

        self.tp = e3nn.o3.FullyConnectedTensorProduct(
            in_irreps,
            sh_irreps,
            out_irreps,
            shared_weights=False,
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(edge_feats_dim, mlp_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(mlp_dim, self.tp.weight_numel),
        )
        self.batch_norm = e3nn.nn.BatchNorm(out_irreps) if batch_norm else None

    def forward(self, node_attr: Tensor, edge_index: Tensor, edge_sh: Tensor, edge_feat: Tensor) -> Tensor:
        src, dst = edge_index
        tp = self.tp(node_attr[dst], edge_sh, self.fc(edge_feat))
        out = scatter(tp, src, dim=0, reduce=self.aggr)
        if self.gate:
            out = self.gate(out)
        if self.batch_norm:
            out = self.batch_norm(out)
        return out


class TFNModel(torch.nn.Module):
    def __init__(
        self,
        max_ell: int = 4,
        num_layer: int = 4,
        hidden_dim: int = 64,
        irreps_channels: int = 16,
        node_input_dim: int = 101,
        edge_input_dim: int = 1,
        r_max: float = 10.0,
        num_bessel: int = 8,
        num_polynomial_cutoff: int = 5,
        hidden_irreps: Optional[e3nn.o3.Irreps] = None,
        mlp_dim: Optional[int] = None,
        aggr: str = "mean",
        pool: str = "mean",
        gate: bool = True,
        batch_norm: bool = False,
        residual: bool = True,
        device: str = "cpu",
        **kwargs,
    ) -> None:
        del edge_input_dim, kwargs
        super().__init__()

        self.hidden_dim = hidden_dim
        self.irreps_channels = irreps_channels
        self.residual = residual
        self.mlp_dim = hidden_dim if mlp_dim is None else mlp_dim

        self.radial_embedding = RadialEmbeddingBlock(
            r_max=r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
        )
        sh_irreps = e3nn.o3.Irreps.spherical_harmonics(max_ell)
        self.spherical_harmonics = e3nn.o3.SphericalHarmonics(
            sh_irreps,
            normalize=True,
            normalization="component",
        )

        self.node_embedding = nn.Linear(node_input_dim, irreps_channels)

        if hidden_irreps is None:
            hidden_irreps = (sh_irreps * irreps_channels).sort()[0].simplify()
        self.hidden_irreps = hidden_irreps
        self.non_scalar_irreps = _non_scalar_irreps(self.hidden_irreps)
        scalar_mul = sum(mul for mul, ir in self.hidden_irreps if ir.l == 0 and ir.p == 1)
        if scalar_mul == 0:
            raise ValueError("TFN hidden_irreps must contain at least one 0e channel.")
        vector_mul = sum(mul for mul, ir in self.non_scalar_irreps if ir.l == 1 and ir.p == -1)
        assert vector_mul > 0, "TFN hidden_irreps must contain at least one 1o channel."
        self.node_scalar_readout = e3nn.o3.Linear(
            self.hidden_irreps,
            e3nn.o3.Irreps(f"{hidden_dim}x0e"),
        )
        self.node_pos_proj = e3nn.o3.Linear(self.hidden_irreps, e3nn.o3.Irreps("1x1o"))

        self.convs = nn.ModuleList()
        self.convs.append(
            TensorProductConvLayer(
                in_irreps=e3nn.o3.Irreps(f"{irreps_channels}x0e"),
                out_irreps=hidden_irreps,
                sh_irreps=sh_irreps,
                edge_feats_dim=self.radial_embedding.out_dim,
                mlp_dim=self.mlp_dim,
                aggr=aggr,
                batch_norm=batch_norm,
                gate=gate,
            )
        )
        for _ in range(num_layer - 1):
            self.convs.append(
                TensorProductConvLayer(
                    in_irreps=hidden_irreps,
                    out_irreps=hidden_irreps,
                    sh_irreps=sh_irreps,
                    edge_feats_dim=self.radial_embedding.out_dim,
                    mlp_dim=self.mlp_dim,
                    aggr=aggr,
                    batch_norm=batch_norm,
                    gate=gate,
                )
            )

        self.scalar_readout = e3nn.o3.Linear(self.hidden_irreps, e3nn.o3.Irreps(f"{hidden_dim}x0e"))
        self.output_dim = hidden_dim
        self.pool = {"mean": global_mean_pool, "sum": global_add_pool}[pool]
        self.to(device)

    def forward(self, data: Data):
        h = self.node_embedding(data.x.float())

        vectors = data.pos[data.edge_index[0]] - data.pos[data.edge_index[1]]
        lengths = torch.linalg.norm(vectors, dim=-1, keepdim=True)

        edge_sh = self.spherical_harmonics(vectors)
        edge_feats = self.radial_embedding(lengths)

        for conv in self.convs:
            h_update = conv(h, data.edge_index, edge_sh, edge_feats)
            h = h_update + F.pad(h, (0, h_update.shape[-1] - h.shape[-1])) if self.residual else h_update

        node_feat = self.node_scalar_readout(h)
        node_sh = h
        node_pos = data.pos + self.node_pos_proj(node_sh)
        graph_irreps = self.pool(h, data.batch)
        graph_feat = self.scalar_readout(graph_irreps)
        graph_sh = graph_irreps
        return build_model_output(
            node_feat=node_feat,
            node_sh=node_sh,
            graph_feat=graph_feat,
            graph_sh=graph_sh,
            node_pos=node_pos,
            node_irreps=h,
            graph_irreps=graph_irreps,
        )
