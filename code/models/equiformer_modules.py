from __future__ import annotations

import collections
import math
from typing import Optional, Sequence

import e3nn
import torch
import torch.nn.functional as F
from e3nn import o3
from e3nn.math import perm
from torch import Tensor, nn
from torch_geometric.utils import degree, softmax
from torch_scatter import scatter


_RESCALE = True
_USE_BIAS = True


def _non_scalar_irreps(irreps: o3.Irreps) -> o3.Irreps:
    parts = [(mul, ir) for mul, ir in irreps if not (ir.l == 0 and ir.p == 1)]
    return o3.Irreps(parts)


def _parity_complete_irreps(max_ell: int, irreps_channels: int) -> o3.Irreps:
    parts = []
    for ell in range(max_ell + 1):
        parts.append((irreps_channels, (ell, 1)))
        parts.append((irreps_channels, (ell, -1)))
    return o3.Irreps(parts).sort()[0].simplify()


def _scale_irreps_for_heads(irreps: o3.Irreps, num_heads: int) -> o3.Irreps:
    if num_heads <= 0:
        raise ValueError(f"num_heads must be positive, got {num_heads}.")
    parts = [(max(1, math.ceil(mul / num_heads)), ir) for mul, ir in irreps]
    return o3.Irreps(parts).sort()[0].simplify()


def get_norm_layer(norm_type: Optional[str]):
    if norm_type in {"layer", "graph", "instance", "fast_layer"}:
        return EquivariantLayerNormV2
    if norm_type is None:
        return None
    raise ValueError(f"Norm type {norm_type} is not supported.")


class SmoothLeakyReLU(nn.Module):
    def __init__(self, negative_slope: float = 0.2):
        super().__init__()
        self.alpha = negative_slope

    def forward(self, x: Tensor) -> Tensor:
        x1 = ((1 + self.alpha) / 2) * x
        x2 = ((1 - self.alpha) / 2) * x * (2 * torch.sigmoid(x) - 1)
        return x1 + x2


def get_mul_0(irreps: o3.Irreps) -> int:
    mul_0 = 0
    for mul, ir in irreps:
        if ir.l == 0 and ir.p == 1:
            mul_0 += mul
    return mul_0


class EquivariantLayerNormV2(nn.Module):
    def __init__(self, irreps, eps: float = 1e-5, affine: bool = True, normalization: str = "component"):
        super().__init__()
        self.irreps = o3.Irreps(irreps)
        self.eps = eps
        self.affine = affine
        self.normalization = normalization

        num_scalar = sum(mul for mul, ir in self.irreps if ir.l == 0 and ir.p == 1)
        num_features = self.irreps.num_irreps
        if affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_scalar))
        else:
            self.register_parameter("affine_weight", None)
            self.register_parameter("affine_bias", None)

        if normalization not in {"norm", "component"}:
            raise ValueError(f"Invalid normalization option {normalization}.")

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, node_input: Tensor, **kwargs) -> Tensor:
        del kwargs
        dim = node_input.shape[-1]

        fields = []
        ix = 0
        iw = 0
        ib = 0

        for mul, ir in self.irreps:
            d = ir.dim
            field = node_input.narrow(1, ix, mul * d).reshape(-1, mul, d)
            ix += mul * d

            if ir.l == 0 and ir.p == 1:
                field = field - torch.mean(field, dim=1, keepdim=True)

            if self.normalization == "norm":
                field_norm = field.pow(2).sum(-1)
            else:
                field_norm = field.pow(2).mean(-1)
            field_norm = torch.mean(field_norm, dim=1, keepdim=True)
            field_norm = (field_norm + self.eps).pow(-0.5)

            if self.affine:
                weight = self.affine_weight[None, iw : iw + mul]
                iw += mul
                field_norm = field_norm * weight

            field = field * field_norm.reshape(-1, mul, 1)

            if self.affine and d == 1 and ir.p == 1:
                bias = self.affine_bias[ib : ib + mul]
                ib += mul
                field = field + bias.reshape(1, mul, 1)

            fields.append(field.reshape(-1, mul * d))

        if ix != dim:
            raise AssertionError(f"`ix` should have reached node_input.size(-1) ({dim}), but it ended at {ix}.")

        return torch.cat(fields, dim=-1)


class GraphDropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor, batch: Tensor) -> Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        batch_size = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
        keep_prob = 1.0 - self.drop_prob
        shape = (batch_size,) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor[batch]


class EquivariantDropout(nn.Module):
    def __init__(self, irreps: o3.Irreps, drop_prob: float):
        super().__init__()
        self.irreps = o3.Irreps(irreps)
        self.num_irreps = self.irreps.num_irreps
        self.drop_prob = drop_prob
        self.mul = o3.ElementwiseTensorProduct(self.irreps, o3.Irreps(f"{self.num_irreps}x0e"))

    def forward(self, x: Tensor) -> Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        mask = torch.ones((x.shape[0], self.num_irreps), dtype=x.dtype, device=x.device)
        mask = F.dropout(mask, p=self.drop_prob, training=True)
        return self.mul(x, mask)


class GaussianRadialBasisLayer(nn.Module):
    def __init__(self, num_basis: int, cutoff: float):
        super().__init__()
        self.num_basis = num_basis
        self.cutoff = float(cutoff)
        self.mean = nn.Parameter(torch.zeros(1, self.num_basis))
        self.std = nn.Parameter(torch.zeros(1, self.num_basis))
        self.weight = nn.Parameter(torch.ones(1, 1))
        self.bias = nn.Parameter(torch.zeros(1, 1))
        nn.init.uniform_(self.mean, 0.0, 1.0)
        nn.init.uniform_(self.std, 1.0 / self.num_basis, 1.0)
        nn.init.constant_(self.weight, 1.0)
        nn.init.constant_(self.bias, 0.0)

    def forward(self, dist: Tensor) -> Tensor:
        x = (dist / self.cutoff).unsqueeze(-1)
        x = self.weight * x + self.bias
        x = x.expand(-1, self.num_basis)
        std = self.std.abs() + 1e-5
        return torch.exp(-0.5 * (((x - self.mean) / std) ** 2)) / (((2 * math.pi) ** 0.5) * std)


class RadialProfile(nn.Module):
    def __init__(self, ch_list: Sequence[int], use_layer_norm: bool = True, use_offset: bool = True):
        super().__init__()
        modules = []
        input_channels = ch_list[0]
        for idx, out_channels in enumerate(ch_list[1:], start=1):
            use_bias = not (idx == len(ch_list) - 1 and use_offset)
            modules.append(nn.Linear(input_channels, out_channels, bias=use_bias))
            input_channels = out_channels
            if idx == len(ch_list) - 1:
                break
            if use_layer_norm:
                modules.append(nn.LayerNorm(out_channels))
            modules.append(nn.SiLU())
        self.net = nn.Sequential(*modules)
        self.offset = None
        if use_offset:
            self.offset = nn.Parameter(torch.zeros(ch_list[-1]))
            fan_in = ch_list[-2]
            bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
            nn.init.uniform_(self.offset, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        out = self.net(x)
        if self.offset is not None:
            out = out + self.offset.reshape(1, -1)
        return out


class TensorProductRescale(nn.Module):
    def __init__(
        self,
        irreps_in1,
        irreps_in2,
        irreps_out,
        instructions,
        bias: bool = True,
        rescale: bool = True,
        internal_weights=None,
        shared_weights=None,
        normalization=None,
    ):
        super().__init__()
        self.irreps_in1 = o3.Irreps(irreps_in1)
        self.irreps_in2 = o3.Irreps(irreps_in2)
        self.irreps_out = o3.Irreps(irreps_out)
        self.rescale = rescale
        self.use_bias = bias
        self.tp = o3.TensorProduct(
            irreps_in1=self.irreps_in1,
            irreps_in2=self.irreps_in2,
            irreps_out=self.irreps_out,
            instructions=instructions,
            normalization=normalization,
            internal_weights=internal_weights,
            shared_weights=shared_weights,
            path_normalization="none",
        )
        self.init_rescale_bias()

    def calculate_fan_in(self, ins) -> int:
        return {
            "uvw": self.irreps_in1[ins.i_in1].mul * self.irreps_in2[ins.i_in2].mul,
            "uvu": self.irreps_in2[ins.i_in2].mul,
            "uvv": self.irreps_in1[ins.i_in1].mul,
            "uuw": self.irreps_in1[ins.i_in1].mul,
            "uuu": 1,
            "uvuv": 1,
            "uvu<v": 1,
            "u<vw": self.irreps_in1[ins.i_in1].mul * (self.irreps_in2[ins.i_in2].mul - 1) // 2,
        }[ins.connection_mode]

    def init_rescale_bias(self) -> None:
        self.bias_slices = []
        self.bias_slice_idx = []
        self.bias = nn.ParameterList()
        self.irreps_bias = self.irreps_out.simplify()
        self.irreps_out_slices = self.irreps_out.slices()
        if self.use_bias:
            for slice_idx, ((mul, ir), out_slice) in enumerate(zip(self.irreps_bias, self.irreps_bias.slices())):
                if ir.l == 0 and ir.p == 1:
                    out_bias = nn.Parameter(torch.zeros(mul, dtype=self.tp.weight.dtype))
                    self.bias.append(out_bias)
                    self.bias_slices.append(out_slice)
                    self.bias_slice_idx.append(slice_idx)

        self.slices_sqrt_k = {}
        with torch.no_grad():
            slices_fan_in = {}
            for instr in self.tp.instructions:
                slice_idx = instr[2]
                fan_in = self.calculate_fan_in(instr)
                slices_fan_in[slice_idx] = slices_fan_in.get(slice_idx, 0) + fan_in
            for instr in self.tp.instructions:
                slice_idx = instr[2]
                sqrt_k = 1.0 / slices_fan_in[slice_idx] ** 0.5 if self.rescale else 1.0
                self.slices_sqrt_k[slice_idx] = (self.irreps_out_slices[slice_idx], sqrt_k)

            if self.tp.internal_weights:
                for weight, instr in zip(self.tp.weight_views(), self.tp.instructions):
                    if self.rescale:
                        slice_idx = instr[2]
                        sqrt_k = 1.0 / slices_fan_in[slice_idx] ** 0.5
                        weight.data.mul_(sqrt_k)

    def forward_tp_rescale_bias(self, x: Tensor, y: Tensor, weight: Optional[Tensor] = None) -> Tensor:
        out = self.tp(x, y, weight)
        if self.use_bias:
            for out_slice, bias in zip(self.bias_slices, self.bias):
                out.narrow(1, out_slice.start, out_slice.stop - out_slice.start).add_(bias)
        return out

    def forward(self, x: Tensor, y: Tensor, weight: Optional[Tensor] = None) -> Tensor:
        return self.forward_tp_rescale_bias(x, y, weight)


class FullyConnectedTensorProductRescale(TensorProductRescale):
    def __init__(
        self,
        irreps_in1,
        irreps_in2,
        irreps_out,
        bias: bool = True,
        rescale: bool = True,
        internal_weights=None,
        shared_weights=None,
        normalization=None,
    ):
        irreps_in1 = o3.Irreps(irreps_in1)
        irreps_in2 = o3.Irreps(irreps_in2)
        irreps_out = o3.Irreps(irreps_out)
        instructions = [
            (i_1, i_2, i_out, "uvw", True, 1.0)
            for i_1, (_, ir_1) in enumerate(irreps_in1)
            for i_2, (_, ir_2) in enumerate(irreps_in2)
            for i_out, (_, ir_out) in enumerate(irreps_out)
            if ir_out in ir_1 * ir_2
        ]
        super().__init__(
            irreps_in1=irreps_in1,
            irreps_in2=irreps_in2,
            irreps_out=irreps_out,
            instructions=instructions,
            bias=bias,
            rescale=rescale,
            internal_weights=internal_weights,
            shared_weights=shared_weights,
            normalization=normalization,
        )


class LinearRS(FullyConnectedTensorProductRescale):
    def __init__(self, irreps_in, irreps_out, bias: bool = True, rescale: bool = True):
        super().__init__(
            irreps_in1=o3.Irreps(irreps_in),
            irreps_in2=o3.Irreps("1x0e"),
            irreps_out=o3.Irreps(irreps_out),
            bias=bias,
            rescale=rescale,
            internal_weights=True,
            shared_weights=True,
            normalization=None,
        )

    def forward(self, x: Tensor) -> Tensor:
        y = torch.ones_like(x[:, :1])
        return self.forward_tp_rescale_bias(x, y)


def irreps2gate(irreps) -> tuple[o3.Irreps, o3.Irreps, o3.Irreps]:
    irreps = o3.Irreps(irreps)
    irreps_scalars = []
    irreps_gated = []
    for mul, ir in irreps:
        if ir.l == 0 and ir.p == 1:
            irreps_scalars.append((mul, ir))
        else:
            irreps_gated.append((mul, ir))
    irreps_scalars = o3.Irreps(irreps_scalars).simplify()
    irreps_gated = o3.Irreps(irreps_gated).simplify()
    gate_ir = "0e" if irreps_gated.dim > 0 else None
    irreps_gates = o3.Irreps([(mul, gate_ir) for mul, _ in irreps_gated]).simplify()
    return irreps_scalars, irreps_gates, irreps_gated


def sort_irreps_even_first(irreps):
    ret = collections.namedtuple("sort", ["irreps", "p", "inv"])
    out = [(ir.l, -ir.p, i, mul) for i, (mul, ir) in enumerate(o3.Irreps(irreps))]
    out = sorted(out)
    inv = tuple(i for _, _, i, _ in out)
    p = perm.inverse(inv)
    sorted_irreps = o3.Irreps([(mul, (l, -parity)) for l, parity, _, mul in out])
    return ret(sorted_irreps, p, inv)


class FullyConnectedTensorProductRescaleNorm(FullyConnectedTensorProductRescale):
    def __init__(
        self,
        irreps_in1,
        irreps_in2,
        irreps_out,
        bias: bool = True,
        rescale: bool = True,
        internal_weights=None,
        shared_weights=None,
        normalization=None,
        norm_layer: Optional[str] = "layer",
    ):
        super().__init__(
            irreps_in1=irreps_in1,
            irreps_in2=irreps_in2,
            irreps_out=irreps_out,
            bias=bias,
            rescale=rescale,
            internal_weights=internal_weights,
            shared_weights=shared_weights,
            normalization=normalization,
        )
        norm = get_norm_layer(norm_layer)
        self.norm = norm(self.irreps_out) if norm is not None else None

    def forward(self, x: Tensor, y: Tensor, batch: Tensor, weight: Optional[Tensor] = None) -> Tensor:
        out = self.forward_tp_rescale_bias(x, y, weight)
        if self.norm is not None:
            out = self.norm(out, batch=batch)
        return out


class FullyConnectedTensorProductRescaleNormSwishGate(FullyConnectedTensorProductRescaleNorm):
    def __init__(
        self,
        irreps_in1,
        irreps_in2,
        irreps_out,
        bias: bool = True,
        rescale: bool = True,
        internal_weights=None,
        shared_weights=None,
        normalization=None,
        norm_layer: Optional[str] = "layer",
    ):
        irreps_scalars, irreps_gates, irreps_gated = irreps2gate(irreps_out)
        if irreps_gated.num_irreps == 0:
            gate = e3nn.nn.Activation(irreps_out, acts=[torch.nn.SiLU()])
        else:
            gate = e3nn.nn.Gate(
                irreps_scalars,
                [torch.nn.SiLU() for _ in irreps_scalars],
                irreps_gates,
                [torch.sigmoid for _ in irreps_gates],
                irreps_gated,
            )
        super().__init__(
            irreps_in1=irreps_in1,
            irreps_in2=irreps_in2,
            irreps_out=gate.irreps_in,
            bias=bias,
            rescale=rescale,
            internal_weights=internal_weights,
            shared_weights=shared_weights,
            normalization=normalization,
            norm_layer=norm_layer,
        )
        self.gate = gate

    def forward(self, x: Tensor, y: Tensor, batch: Tensor, weight: Optional[Tensor] = None) -> Tensor:
        out = super().forward(x, y, batch=batch, weight=weight)
        return self.gate(out)


class FullyConnectedTensorProductRescaleSwishGate(FullyConnectedTensorProductRescale):
    def __init__(
        self,
        irreps_in1,
        irreps_in2,
        irreps_out,
        bias: bool = True,
        rescale: bool = True,
        internal_weights=None,
        shared_weights=None,
        normalization=None,
    ):
        irreps_scalars, irreps_gates, irreps_gated = irreps2gate(irreps_out)
        if irreps_gated.num_irreps == 0:
            gate = e3nn.nn.Activation(irreps_out, acts=[torch.nn.SiLU()])
        else:
            gate = e3nn.nn.Gate(
                irreps_scalars,
                [torch.nn.SiLU() for _ in irreps_scalars],
                irreps_gates,
                [torch.sigmoid for _ in irreps_gates],
                irreps_gated,
            )
        super().__init__(
            irreps_in1=irreps_in1,
            irreps_in2=irreps_in2,
            irreps_out=gate.irreps_in,
            bias=bias,
            rescale=rescale,
            internal_weights=internal_weights,
            shared_weights=shared_weights,
            normalization=normalization,
        )
        self.gate = gate

    def forward(self, x: Tensor, y: Tensor, weight: Optional[Tensor] = None) -> Tensor:
        return self.gate(super().forward(x, y, weight=weight))


def DepthwiseTensorProduct(irreps_node_input, irreps_edge_attr, irreps_node_output, internal_weights=False, bias=True):
    irreps_node_input = o3.Irreps(irreps_node_input)
    irreps_edge_attr = o3.Irreps(irreps_edge_attr)
    irreps_node_output = o3.Irreps(irreps_node_output)

    irreps_output = []
    instructions = []
    for i, (mul, ir_in) in enumerate(irreps_node_input):
        for j, (_, ir_edge) in enumerate(irreps_edge_attr):
            for ir_out in ir_in * ir_edge:
                if ir_out in irreps_node_output or ir_out == o3.Irrep(0, 1):
                    k = len(irreps_output)
                    irreps_output.append((mul, ir_out))
                    instructions.append((i, j, k, "uvu", True))

    irreps_output = o3.Irreps(irreps_output)
    irreps_output, p, _ = sort_irreps_even_first(irreps_output)
    instructions = [(i_1, i_2, p[i_out], mode, train) for i_1, i_2, i_out, mode, train in instructions]
    return TensorProductRescale(
        irreps_in1=irreps_node_input,
        irreps_in2=irreps_edge_attr,
        irreps_out=irreps_output,
        instructions=instructions,
        internal_weights=internal_weights,
        shared_weights=internal_weights,
        bias=bias,
        rescale=_RESCALE,
    )


class SeparableFCTP(nn.Module):
    def __init__(
        self,
        irreps_node_input,
        irreps_edge_attr,
        irreps_node_output,
        fc_neurons: Optional[Sequence[int]],
        use_activation: bool = False,
        norm_layer: Optional[str] = "layer",
        internal_weights: bool = False,
    ):
        super().__init__()
        self.irreps_node_input = o3.Irreps(irreps_node_input)
        self.irreps_edge_attr = o3.Irreps(irreps_edge_attr)
        self.irreps_node_output = o3.Irreps(irreps_node_output)

        self.dtp = DepthwiseTensorProduct(
            self.irreps_node_input,
            self.irreps_edge_attr,
            self.irreps_node_output,
            bias=False,
            internal_weights=internal_weights,
        )

        self.dtp_rad = None
        if fc_neurons is not None:
            fc_neurons = list(fc_neurons)
            self.dtp_rad = RadialProfile(fc_neurons + [self.dtp.tp.weight_numel])
            for out_slice, slice_sqrt_k in self.dtp.slices_sqrt_k.values():
                self.dtp_rad.net[-1].weight.data[out_slice, :] *= slice_sqrt_k
                self.dtp_rad.offset.data[out_slice] *= slice_sqrt_k

        irreps_lin_output = self.irreps_node_output
        irreps_scalars, irreps_gates, irreps_gated = irreps2gate(self.irreps_node_output)
        if use_activation:
            irreps_lin_output = (irreps_scalars + irreps_gates + irreps_gated).simplify()
        self.lin = LinearRS(self.dtp.irreps_out.simplify(), irreps_lin_output)

        norm = get_norm_layer(norm_layer)
        self.norm = norm(self.lin.irreps_out) if norm is not None else None

        self.gate = None
        if use_activation:
            if irreps_gated.num_irreps == 0:
                self.gate = e3nn.nn.Activation(self.irreps_node_output, acts=[torch.nn.SiLU()])
            else:
                self.gate = e3nn.nn.Gate(
                    irreps_scalars,
                    [torch.nn.SiLU() for _ in irreps_scalars],
                    irreps_gates,
                    [torch.sigmoid for _ in irreps_gates],
                    irreps_gated,
                )

    def forward(self, node_input: Tensor, edge_attr: Tensor, edge_scalars: Optional[Tensor], batch: Optional[Tensor] = None):
        weight = self.dtp_rad(edge_scalars) if self.dtp_rad is not None and edge_scalars is not None else None
        out = self.dtp(node_input, edge_attr, weight)
        out = self.lin(out)
        if self.norm is not None:
            out = self.norm(out, batch=batch)
        if self.gate is not None:
            out = self.gate(out)
        return out


class Vec2AttnHeads(nn.Module):
    def __init__(self, irreps_head, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.irreps_head = o3.Irreps(irreps_head)
        self.irreps_mid_in = o3.Irreps([(mul * num_heads, ir) for mul, ir in self.irreps_head])
        self.mid_in_indices = []
        start_idx = 0
        for mul, ir in self.irreps_mid_in:
            self.mid_in_indices.append((start_idx, start_idx + mul * ir.dim))
            start_idx += mul * ir.dim

    def forward(self, x: Tensor) -> Tensor:
        out = []
        for start_idx, end_idx in self.mid_in_indices:
            chunk = x.narrow(1, start_idx, end_idx - start_idx)
            out.append(chunk.reshape(x.shape[0], self.num_heads, -1))
        return torch.cat(out, dim=2)


class AttnHeads2Vec(nn.Module):
    def __init__(self, irreps_head):
        super().__init__()
        self.irreps_head = o3.Irreps(irreps_head)
        self.head_indices = []
        start_idx = 0
        for mul, ir in self.irreps_head:
            self.head_indices.append((start_idx, start_idx + mul * ir.dim))
            start_idx += mul * ir.dim

    def forward(self, x: Tensor) -> Tensor:
        out = []
        for start_idx, end_idx in self.head_indices:
            chunk = x.narrow(2, start_idx, end_idx - start_idx)
            out.append(chunk.reshape(x.shape[0], -1))
        return torch.cat(out, dim=1)


class GraphAttention(nn.Module):
    def __init__(
        self,
        irreps_node_input,
        irreps_node_attr,
        irreps_edge_attr,
        irreps_node_output,
        fc_neurons: Sequence[int],
        irreps_head,
        num_heads: int,
        irreps_pre_attn=None,
        rescale_degree: bool = False,
        nonlinear_message: bool = False,
        alpha_drop: float = 0.1,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.irreps_node_input = o3.Irreps(irreps_node_input)
        self.irreps_node_attr = o3.Irreps(irreps_node_attr)
        self.irreps_edge_attr = o3.Irreps(irreps_edge_attr)
        self.irreps_node_output = o3.Irreps(irreps_node_output)
        self.irreps_pre_attn = self.irreps_node_input if irreps_pre_attn is None else o3.Irreps(irreps_pre_attn)
        self.irreps_head = o3.Irreps(irreps_head)
        self.num_heads = num_heads
        self.rescale_degree = rescale_degree
        self.nonlinear_message = nonlinear_message

        self.merge_src = LinearRS(self.irreps_node_input, self.irreps_pre_attn, bias=True)
        self.merge_dst = LinearRS(self.irreps_node_input, self.irreps_pre_attn, bias=False)

        irreps_attn_heads = (self.irreps_head * self.num_heads).sort()[0].simplify()
        mul_alpha = get_mul_0(irreps_attn_heads)
        if mul_alpha <= 0 or mul_alpha % self.num_heads != 0:
            raise ValueError("Equiformer attention requires at least one 0e channel per head.")
        mul_alpha_head = mul_alpha // self.num_heads
        irreps_alpha = o3.Irreps(f"{mul_alpha}x0e")
        irreps_attn_all = (irreps_alpha + irreps_attn_heads).simplify()

        self.sep_act = None
        if self.nonlinear_message:
            self.sep_act = SeparableFCTP(
                self.irreps_pre_attn,
                self.irreps_edge_attr,
                self.irreps_pre_attn,
                fc_neurons=fc_neurons,
                use_activation=True,
                norm_layer=None,
                internal_weights=False,
            )
            self.sep_alpha = LinearRS(self.sep_act.dtp.irreps_out, irreps_alpha)
            self.sep_value = SeparableFCTP(
                self.irreps_pre_attn,
                self.irreps_edge_attr,
                irreps_attn_heads,
                fc_neurons=None,
                use_activation=False,
                norm_layer=None,
                internal_weights=True,
            )
            self.vec2heads_alpha = Vec2AttnHeads(o3.Irreps(f"{mul_alpha_head}x0e"), self.num_heads)
            self.vec2heads_value = Vec2AttnHeads(self.irreps_head, self.num_heads)
        else:
            self.sep = SeparableFCTP(
                self.irreps_pre_attn,
                self.irreps_edge_attr,
                irreps_attn_all,
                fc_neurons=fc_neurons,
                use_activation=False,
                norm_layer=None,
            )
            self.vec2heads = Vec2AttnHeads((o3.Irreps(f"{mul_alpha_head}x0e") + self.irreps_head).simplify(), self.num_heads)

        self.alpha_act = e3nn.nn.Activation(o3.Irreps(f"{mul_alpha_head}x0e"), [SmoothLeakyReLU(0.2)])
        self.heads2vec = AttnHeads2Vec(self.irreps_head)
        self.mul_alpha_head = mul_alpha_head
        self.alpha_dot = nn.Parameter(torch.empty(1, self.num_heads, self.mul_alpha_head))
        nn.init.xavier_uniform_(self.alpha_dot)
        self.alpha_dropout = nn.Dropout(alpha_drop) if alpha_drop != 0.0 else None
        self.proj = LinearRS(irreps_attn_heads, self.irreps_node_output)
        self.proj_drop = EquivariantDropout(self.irreps_node_output, proj_drop) if proj_drop != 0.0 else None

    def forward(
        self,
        node_input: Tensor,
        node_attr: Tensor,
        edge_src: Tensor,
        edge_dst: Tensor,
        edge_attr: Tensor,
        edge_scalars: Tensor,
        batch: Tensor,
    ) -> Tensor:
        del node_attr, batch
        message = self.merge_src(node_input)[edge_src] + self.merge_dst(node_input)[edge_dst]

        if self.nonlinear_message:
            weight = self.sep_act.dtp_rad(edge_scalars)
            message = self.sep_act.dtp(message, edge_attr, weight)
            alpha = self.vec2heads_alpha(self.sep_alpha(message))
            value = self.sep_act.gate(self.sep_act.lin(message))
            value = self.vec2heads_value(self.sep_value(value, edge_attr=edge_attr, edge_scalars=edge_scalars))
        else:
            message = self.vec2heads(self.sep(message, edge_attr=edge_attr, edge_scalars=edge_scalars))
            head_dim = message.shape[-1]
            alpha = message.narrow(2, 0, self.mul_alpha_head)
            value = message.narrow(2, self.mul_alpha_head, head_dim - self.mul_alpha_head)

        alpha = self.alpha_act(alpha)
        alpha = torch.einsum("ehk,ahk->eh", alpha, self.alpha_dot)
        alpha = softmax(alpha, edge_dst).unsqueeze(-1)
        if self.alpha_dropout is not None:
            alpha = self.alpha_dropout(alpha)
        attn = value * alpha
        attn = scatter(attn, index=edge_dst, dim=0, dim_size=node_input.shape[0], reduce="sum")
        attn = self.heads2vec(attn)

        if self.rescale_degree:
            deg = degree(edge_dst, num_nodes=node_input.shape[0], dtype=node_input.dtype).view(-1, 1)
            attn = attn * deg

        out = self.proj(attn)
        if self.proj_drop is not None:
            out = self.proj_drop(out)
        return out


class FeedForwardNetwork(nn.Module):
    def __init__(self, irreps_node_input, irreps_node_attr, irreps_node_output, irreps_mlp_mid=None, proj_drop: float = 0.0):
        super().__init__()
        self.irreps_node_input = o3.Irreps(irreps_node_input)
        self.irreps_node_attr = o3.Irreps(irreps_node_attr)
        self.irreps_mlp_mid = self.irreps_node_input if irreps_mlp_mid is None else o3.Irreps(irreps_mlp_mid)
        self.irreps_node_output = o3.Irreps(irreps_node_output)

        self.fctp_1 = FullyConnectedTensorProductRescaleSwishGate(
            self.irreps_node_input,
            self.irreps_node_attr,
            self.irreps_mlp_mid,
            bias=True,
            rescale=_RESCALE,
        )
        self.fctp_2 = FullyConnectedTensorProductRescale(
            self.irreps_mlp_mid,
            self.irreps_node_attr,
            self.irreps_node_output,
            bias=True,
            rescale=_RESCALE,
        )
        self.proj_drop = EquivariantDropout(self.irreps_node_output, proj_drop) if proj_drop != 0.0 else None

    def forward(self, node_input: Tensor, node_attr: Tensor) -> Tensor:
        node_output = self.fctp_1(node_input, node_attr)
        node_output = self.fctp_2(node_output, node_attr)
        if self.proj_drop is not None:
            node_output = self.proj_drop(node_output)
        return node_output


class TransBlock(nn.Module):
    def __init__(
        self,
        irreps_node_input,
        irreps_node_attr,
        irreps_edge_attr,
        irreps_node_output,
        fc_neurons: Sequence[int],
        irreps_head,
        num_heads: int,
        irreps_pre_attn=None,
        rescale_degree: bool = False,
        nonlinear_message: bool = False,
        alpha_drop: float = 0.0,
        proj_drop: float = 0.0,
        drop_path_rate: float = 0.0,
        irreps_mlp_mid=None,
        norm_layer: Optional[str] = "layer",
    ):
        super().__init__()
        self.irreps_node_input = o3.Irreps(irreps_node_input)
        self.irreps_node_attr = o3.Irreps(irreps_node_attr)
        self.irreps_edge_attr = o3.Irreps(irreps_edge_attr)
        self.irreps_node_output = o3.Irreps(irreps_node_output)
        self.irreps_pre_attn = self.irreps_node_input if irreps_pre_attn is None else o3.Irreps(irreps_pre_attn)
        self.irreps_head = o3.Irreps(irreps_head)
        self.num_heads = num_heads
        self.irreps_mlp_mid = self.irreps_node_input if irreps_mlp_mid is None else o3.Irreps(irreps_mlp_mid)

        norm = get_norm_layer(norm_layer)
        self.norm_1 = norm(self.irreps_node_input) if norm is not None else nn.Identity()
        self.ga = GraphAttention(
            irreps_node_input=self.irreps_node_input,
            irreps_node_attr=self.irreps_node_attr,
            irreps_edge_attr=self.irreps_edge_attr,
            irreps_node_output=self.irreps_node_input,
            fc_neurons=fc_neurons,
            irreps_head=self.irreps_head,
            num_heads=self.num_heads,
            irreps_pre_attn=self.irreps_pre_attn,
            rescale_degree=rescale_degree,
            nonlinear_message=nonlinear_message,
            alpha_drop=alpha_drop,
            proj_drop=proj_drop,
        )
        self.drop_path = GraphDropPath(drop_path_rate) if drop_path_rate > 0.0 else None
        self.norm_2 = norm(self.irreps_node_input) if norm is not None else nn.Identity()
        self.ffn = FeedForwardNetwork(
            irreps_node_input=self.irreps_node_input,
            irreps_node_attr=self.irreps_node_attr,
            irreps_node_output=self.irreps_node_output,
            irreps_mlp_mid=self.irreps_mlp_mid,
            proj_drop=proj_drop,
        )
        self.ffn_shortcut = None
        if self.irreps_node_input != self.irreps_node_output:
            self.ffn_shortcut = FullyConnectedTensorProductRescale(
                self.irreps_node_input,
                self.irreps_node_attr,
                self.irreps_node_output,
                bias=True,
                rescale=_RESCALE,
            )

    def forward(
        self,
        node_input: Tensor,
        node_attr: Tensor,
        edge_src: Tensor,
        edge_dst: Tensor,
        edge_attr: Tensor,
        edge_scalars: Tensor,
        batch: Tensor,
    ) -> Tensor:
        node_output = node_input
        node_features = self.norm_1(node_input, batch=batch)
        node_features = self.ga(
            node_input=node_features,
            node_attr=node_attr,
            edge_src=edge_src,
            edge_dst=edge_dst,
            edge_attr=edge_attr,
            edge_scalars=edge_scalars,
            batch=batch,
        )
        if self.drop_path is not None:
            node_features = self.drop_path(node_features, batch)
        node_output = node_output + node_features

        node_features = self.norm_2(node_output, batch=batch)
        node_features = self.ffn(node_features, node_attr)
        if self.ffn_shortcut is not None:
            node_output = self.ffn_shortcut(node_output, node_attr)
        if self.drop_path is not None:
            node_features = self.drop_path(node_features, batch)
        node_output = node_output + node_features
        return node_output


class ScaledScatter(nn.Module):
    def __init__(self, avg_aggregate_num: float):
        super().__init__()
        self.avg_aggregate_num = float(avg_aggregate_num)

    def forward(self, x: Tensor, index: Tensor, **kwargs) -> Tensor:
        out = scatter(x, index, **kwargs)
        return out.div(self.avg_aggregate_num ** 0.5)


class EdgeDegreeEmbeddingNetwork(nn.Module):
    def __init__(self, irreps_node_embedding, irreps_edge_attr, fc_neurons: Sequence[int], avg_aggregate_num: float):
        super().__init__()
        self.exp = LinearRS(o3.Irreps("1x0e"), irreps_node_embedding, bias=_USE_BIAS, rescale=_RESCALE)
        self.dw = DepthwiseTensorProduct(irreps_node_embedding, irreps_edge_attr, irreps_node_embedding, internal_weights=False, bias=False)
        self.rad = RadialProfile(list(fc_neurons) + [self.dw.tp.weight_numel])
        for out_slice, slice_sqrt_k in self.dw.slices_sqrt_k.values():
            self.rad.net[-1].weight.data[out_slice, :] *= slice_sqrt_k
            self.rad.offset.data[out_slice] *= slice_sqrt_k
        self.proj = LinearRS(self.dw.irreps_out.simplify(), irreps_node_embedding)
        self.scale_scatter = ScaledScatter(avg_aggregate_num)

    def forward(
        self,
        node_input: Tensor,
        edge_attr: Tensor,
        edge_scalars: Tensor,
        edge_src: Tensor,
        edge_dst: Tensor,
        batch: Tensor,
    ) -> Tensor:
        del batch
        node_features = self.exp(torch.ones_like(node_input[:, :1]))
        weight = self.rad(edge_scalars)
        edge_features = self.dw(node_features[edge_src], edge_attr, weight)
        edge_features = self.proj(edge_features)
        return self.scale_scatter(edge_features, edge_dst, dim=0, dim_size=node_input.shape[0], reduce="sum")


__all__ = [
    "_non_scalar_irreps",
    "_parity_complete_irreps",
    "_scale_irreps_for_heads",
    "get_norm_layer",
    "GaussianRadialBasisLayer",
    "LinearRS",
    "TransBlock",
    "EdgeDegreeEmbeddingNetwork",
]
