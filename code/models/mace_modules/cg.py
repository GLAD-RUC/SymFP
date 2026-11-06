import collections
from typing import List, Union

import torch
from e3nn import o3


_TP = collections.namedtuple("_TP", "op, args")
_INPUT = collections.namedtuple("_INPUT", "tensor, start, stop")
_WIGNER_NJ_CACHE = {}
_U_MATRIX_REAL_CACHE = {}


def _normalize_filter_key(filter_ir_mid):
    if filter_ir_mid is None:
        return None
    return tuple(str(o3.Irrep(ir)) for ir in filter_ir_mid)


def _normalize_irreps_key(irrepss):
    return tuple(str(o3.Irreps(irreps)) for irreps in irrepss)


def _normalize_dtype_key(dtype):
    return None if dtype is None else str(dtype)


def _wigner_nj(
    irrepss: List[o3.Irreps],
    normalization: str = "component",
    filter_ir_mid=None,
    dtype=None,
):
    irrepss = [o3.Irreps(irreps) for irreps in irrepss]
    if filter_ir_mid is not None:
        filter_ir_mid = [o3.Irrep(ir) for ir in filter_ir_mid]

    cache_key = (
        _normalize_irreps_key(irrepss),
        normalization,
        _normalize_filter_key(filter_ir_mid),
        _normalize_dtype_key(dtype),
    )
    cached = _WIGNER_NJ_CACHE.get(cache_key)
    if cached is not None:
        return cached

    if len(irrepss) == 1:
        (irreps,) = irrepss
        ret = []
        eye = torch.eye(irreps.dim, dtype=dtype)
        idx = 0
        for mul, ir in irreps:
            for _ in range(mul):
                sl = slice(idx, idx + ir.dim)
                ret.append((ir, _INPUT(0, sl.start, sl.stop), eye[sl]))
                idx += ir.dim
        _WIGNER_NJ_CACHE[cache_key] = ret
        return ret

    *irrepss_left, irreps_right = irrepss
    ret = []
    for ir_left, path_left, c_left in _wigner_nj(
        irrepss_left,
        normalization=normalization,
        filter_ir_mid=filter_ir_mid,
        dtype=dtype,
    ):
        idx = 0
        for mul, ir in irreps_right:
            for ir_out in ir_left * ir:
                if filter_ir_mid is not None and ir_out not in filter_ir_mid:
                    continue

                c = o3.wigner_3j(ir_out.l, ir_left.l, ir.l, dtype=dtype)
                if normalization == "component":
                    c *= ir_out.dim ** 0.5
                if normalization == "norm":
                    c *= ir_left.dim ** 0.5 * ir.dim ** 0.5

                c = torch.einsum("jk,ijl->ikl", c_left.flatten(1), c)
                c = c.reshape(ir_out.dim, *(irreps.dim for irreps in irrepss_left), ir.dim)
                for u in range(mul):
                    e = torch.zeros(
                        ir_out.dim,
                        *(irreps.dim for irreps in irrepss_left),
                        irreps_right.dim,
                        dtype=dtype,
                    )
                    sl = slice(idx + u * ir.dim, idx + (u + 1) * ir.dim)
                    e[..., sl] = c
                    ret.append(
                        (
                            ir_out,
                            _TP(
                                op=(ir_left, ir, ir_out),
                                args=(path_left, _INPUT(len(irrepss_left), sl.start, sl.stop)),
                            ),
                            e,
                        )
                    )
            idx += mul * ir.dim
    ret = sorted(ret, key=lambda x: x[0])
    _WIGNER_NJ_CACHE[cache_key] = ret
    return ret


def U_matrix_real(
    irreps_in: Union[str, o3.Irreps],
    irreps_out: Union[str, o3.Irreps],
    correlation: int,
    normalization: str = "component",
    filter_ir_mid=None,
    dtype=None,
):
    irreps_in = o3.Irreps(irreps_in)
    irreps_out = o3.Irreps(irreps_out)
    irrepss = [irreps_in] * correlation
    if correlation == 4:
        filter_ir_mid = [
            (0, 1),
            (1, -1),
            (2, 1),
            (3, -1),
            (4, 1),
            (5, -1),
            (6, 1),
            (7, -1),
            (8, 1),
            (9, -1),
            (10, 1),
            (11, -1),
        ]

    cache_key = (
        str(irreps_in),
        str(irreps_out),
        correlation,
        normalization,
        _normalize_filter_key(filter_ir_mid),
        _normalize_dtype_key(dtype),
    )
    cached = _U_MATRIX_REAL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    wigners = _wigner_nj(irrepss, normalization, filter_ir_mid, dtype)
    current_ir = wigners[0][0]
    out = []
    stack = torch.tensor([])

    for ir, _, base_o3 in wigners:
        if ir in irreps_out and ir == current_ir:
            stack = torch.cat((stack, base_o3.squeeze().unsqueeze(-1)), dim=-1)
            last_ir = current_ir
        elif ir in irreps_out and ir != current_ir:
            if len(stack) != 0:
                out += [last_ir, stack]
            stack = base_o3.squeeze().unsqueeze(-1)
            current_ir, last_ir = ir, ir
        else:
            current_ir = ir
    out += [last_ir, stack]
    _U_MATRIX_REAL_CACHE[cache_key] = out
    return out
