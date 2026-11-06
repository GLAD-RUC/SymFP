from __future__ import annotations

import hashlib
from typing import List, Literal, Sequence

import torch


POINT_GROUPS = [
    "1",
    "-1",
    "2",
    "m",
    "2/m",
    "222",
    "mm2",
    "mmm",
    "4",
    "-4",
    "4/m",
    "422",
    "4mm",
    "-42m",
    "4/mmm",
    "3",
    "-3",
    "32",
    "3m",
    "-3m",
    "6",
    "-6",
    "6/m",
    "622",
    "6mm",
    "-6m2",
    "6/mmm",
    "23",
    "m-3",
    "432",
    "-43m",
    "m-3m",
]
POINT_GROUP_TO_IDX = {point_group: idx for idx, point_group in enumerate(POINT_GROUPS)}
ONE_HOT_MODELS = {
    "equiformer",
    "equiformer_balance",
    "hegnn",
    "hegnn_balance",
    "schnet",
    "dimenet",
    "gvp",
    "tfn",
    "tfn_balance",
    "mace",
    "mace_balance",
}
HASH_METHOD = Literal["sha256", "weighted_utf8_sum"]


def parse_noise_levels(text: str) -> List[float]:
    values = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        alpha = float(chunk)
        if alpha < 0:
            raise ValueError(f"Noise level must be non-negative, got {alpha}.")
        values.append(alpha)
    if not values:
        values = [0.0]
    return sorted(set(round(alpha, 6) for alpha in values))


def format_noise_tag(alpha: float) -> str:
    return f"{alpha:.2f}".replace(".", "p")


def build_test_metric_names(
    test_noise_levels: Sequence[float],
    *,
    include_clean: bool = True,
) -> List[str]:
    names: List[str] = ["clean"] if include_clean else []
    for alpha in test_noise_levels:
        if alpha <= 0:
            if not include_clean:
                names.append("clean")
            continue
        names.append(f"noise_{format_noise_tag(alpha)}")
    return names


def stable_int_hash(
    text: str,
    *,
    method: HASH_METHOD = "sha256",
) -> int:
    if method == "sha256":
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return int(digest[:16], 16) % (2**31)
    if method == "weighted_utf8_sum":
        values = torch.tensor(list(text.encode("utf-8")), dtype=torch.int64)
        if values.numel() == 0:
            return 0
        weights = torch.arange(1, values.numel() + 1, dtype=torch.int64)
        return int(((values * weights).sum().item()) % (2**31))
    raise ValueError(f"Unsupported hash method: {method}")


def center_positions(pos: torch.Tensor) -> torch.Tensor:
    pos = pos.to(dtype=torch.float32)
    return pos - pos.mean(dim=0, keepdim=True)


def median_nearest_neighbor_distance(pos: torch.Tensor) -> torch.Tensor:
    if pos.size(0) <= 1:
        return pos.new_tensor(1.0)

    dist = torch.cdist(pos, pos, p=2)
    diag = torch.arange(pos.size(0), device=pos.device)
    dist[diag, diag] = torch.inf
    nn = dist.min(dim=1).values
    nn = nn[torch.isfinite(nn)]
    if nn.numel() == 0:
        return pos.new_tensor(1.0)
    return torch.median(nn).clamp(min=1e-6)


def sample_deterministic_noise(
    shape: torch.Size,
    seed: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    generator = torch.Generator()
    generator.manual_seed(seed)
    noise = torch.randn(shape, generator=generator, dtype=dtype)
    return noise.to(device=device)


def add_scaled_gaussian_noise(
    pos: torch.Tensor,
    *,
    sample_key: str,
    split: str,
    noise_alpha: float,
    noise_seed: int,
    deterministic_noise: bool,
    hash_method: HASH_METHOD = "sha256",
    center: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    pos = center_positions(pos) if center else pos.to(dtype=torch.float32)
    if noise_alpha <= 0:
        return pos, pos.new_tensor(1.0)

    scale = median_nearest_neighbor_distance(pos)
    if deterministic_noise:
        seed = stable_int_hash(
            f"{sample_key}|{split}|{noise_alpha:.6f}|{noise_seed}",
            method=hash_method,
        )
        noise = sample_deterministic_noise(pos.shape, seed, pos.dtype, pos.device)
    else:
        noise = torch.randn_like(pos)

    noisy = pos + (noise_alpha * scale) * noise
    if center:
        noisy = center_positions(noisy)
    return noisy, scale


def build_distance_edge_attr(pos: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    row, col = edge_index
    return torch.linalg.norm(pos[col] - pos[row], dim=-1, keepdim=True)


def build_node_features(
    atomic_numbers: torch.Tensor,
    model_type: str,
    *,
    num_elements: int = 101,
) -> torch.Tensor:
    atomic_numbers = atomic_numbers.to(dtype=torch.long)
    if model_type in ONE_HOT_MODELS:
        features = torch.zeros(atomic_numbers.size(0), num_elements, dtype=torch.float32)
        features.scatter_(1, atomic_numbers.unsqueeze(1), 1.0)
        return features
    return atomic_numbers.unsqueeze(-1).float()


__all__ = [
    "ONE_HOT_MODELS",
    "POINT_GROUPS",
    "POINT_GROUP_TO_IDX",
    "add_scaled_gaussian_noise",
    "build_distance_edge_attr",
    "build_node_features",
    "build_test_metric_names",
    "center_positions",
    "format_noise_tag",
    "median_nearest_neighbor_distance",
    "parse_noise_levels",
    "sample_deterministic_noise",
    "stable_int_hash",
]

