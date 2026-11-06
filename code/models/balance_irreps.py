import e3nn


def sh_total_dim(max_ell: int, irreps_channels: int) -> int:
    sh_irreps = e3nn.o3.Irreps.spherical_harmonics(max_ell)
    return irreps_channels * sh_irreps.dim


def _resolve_irreps_channels(
    max_ell: int,
    irreps_channels: int | None = None,
    total_dim_budget: int | None = None,
) -> int:
    if irreps_channels is not None:
        if irreps_channels <= 0:
            raise ValueError(f"irreps_channels must be positive, got {irreps_channels}.")
        return irreps_channels

    if total_dim_budget is None:
        raise TypeError("Either irreps_channels or total_dim_budget must be provided.")

    sh_dim = e3nn.o3.Irreps.spherical_harmonics(max_ell).dim
    if total_dim_budget <= 0 or total_dim_budget % sh_dim != 0:
        raise ValueError(
            f"total_dim_budget must be a positive multiple of spherical harmonic dim {sh_dim}, "
            f"got {total_dim_budget}."
        )
    return total_dim_budget // sh_dim


def parity_complete_irreps(max_ell: int, irreps_channels: int) -> e3nn.o3.Irreps:
    irreps_channels = _resolve_irreps_channels(max_ell=max_ell, irreps_channels=irreps_channels)
    parts = []
    for ell in range(max_ell + 1):
        parts.append((irreps_channels, (ell, 1)))
        parts.append((irreps_channels, (ell, -1)))
    return e3nn.o3.Irreps(parts).sort()[0].simplify()


def expanded_parity_complete_irreps(max_ell: int, irreps_channels: int) -> e3nn.o3.Irreps:
    irreps_channels = _resolve_irreps_channels(max_ell=max_ell, irreps_channels=irreps_channels)
    parts = []
    for ell in range(max_ell + 1):
        for _ in range(irreps_channels):
            parts.append((1, (ell, 1)))
            parts.append((1, (ell, -1)))
    return e3nn.o3.Irreps(parts).sort()[0]


# Backward-compatible aliases for earlier balance helper names.
def balanced_sh_irreps(
    max_ell: int,
    irreps_channels: int | None = None,
    total_dim_budget: int | None = None,
) -> e3nn.o3.Irreps:
    return parity_complete_irreps(
        max_ell=max_ell,
        irreps_channels=_resolve_irreps_channels(
            max_ell=max_ell,
            irreps_channels=irreps_channels,
            total_dim_budget=total_dim_budget,
        ),
    )


def expanded_balanced_sh_irreps(
    max_ell: int,
    irreps_channels: int | None = None,
    total_dim_budget: int | None = None,
) -> e3nn.o3.Irreps:
    return expanded_parity_complete_irreps(
        max_ell=max_ell,
        irreps_channels=_resolve_irreps_channels(
            max_ell=max_ell,
            irreps_channels=irreps_channels,
            total_dim_budget=total_dim_budget,
        ),
    )
