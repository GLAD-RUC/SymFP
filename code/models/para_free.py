from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from e3nn.o3 import Irreps, SphericalHarmonics
from torch import Tensor, nn
from torch_geometric.data import Data
from torch_scatter import scatter

from .utils import build_model_output


class ParaFreeModel(nn.Module):
    """Deterministic SymFP backbone with node, edge, and hybrid graph-level branches."""

    BRANCH_MODES = ("node", "edge", "hybrid")

    def __init__(
        self,
        max_ell: int = 6,
        num_layer: int = 0,
        hidden_dim: int = 0,
        irreps_channels: int = 16,
        node_input_dim: int = 1,
        edge_input_dim: int = 1,
        activation: Optional[nn.Module] = None,
        norm: Optional[nn.Module] = None,
        eps: float = 1e-10,
        radial_sigma: Optional[float] = None,
        component_mode: str = "none",
        body_local_scale: float = 1.0,
        water_local_scale: float = 1.0,
        relative_vector_scale: float = 1.0,
        branch_mode: str = "hybrid",
        device: str = "cpu",
    ) -> None:
        super().__init__()
        del num_layer, hidden_dim, activation, norm

        if max_ell < 0:
            raise ValueError(f"max_ell must be non-negative, got {max_ell}.")
        if irreps_channels <= 0:
            raise ValueError(f"irreps_channels must be positive, got {irreps_channels}.")
        if node_input_dim <= 0:
            raise ValueError(f"node_input_dim must be positive, got {node_input_dim}.")
        if edge_input_dim < 0:
            raise ValueError(f"edge_input_dim must be non-negative, got {edge_input_dim}.")

        self.max_ell = int(max_ell)
        self.num_radial_basis = int(irreps_channels)
        self.node_attr_dim = int(node_input_dim)
        self.edge_attr_dim = int(edge_input_dim)
        self.eps = float(eps)
        self.component_mode = str(component_mode)
        self.branch_mode = str(branch_mode)

        if self.branch_mode not in self.BRANCH_MODES:
            raise ValueError(
                f"branch_mode must be one of {self.BRANCH_MODES}, got {self.branch_mode!r}."
            )
        if self.component_mode not in {"none", "water_body_relative_centroid"}:
            raise ValueError(
                "component_mode must be one of {'none', 'water_body_relative_centroid'}, "
                f"got {self.component_mode!r}."
            )

        self.body_local_scale = float(body_local_scale)
        self.water_local_scale = float(water_local_scale)
        self.relative_vector_scale = float(relative_vector_scale)
        self.radial_sigma = (
            float(radial_sigma)
            if radial_sigma is not None
            else 1.0 / float(self.num_radial_basis)
        )
        # Relative to the mean radius, below which a node counts as being at the centre.
        self.center_tol = 1e-5

        self.node_channels = self.node_attr_dim * self.num_radial_basis
        self.edge_channels = (2 * self.node_attr_dim + self.edge_attr_dim) * self.num_radial_basis
        self.hybrid_channels = self.node_channels + self.edge_channels

        self.sh_irreps = Irreps.spherical_harmonics(self.max_ell)
        self.node_hidden_irreps = (self.sh_irreps * self.node_channels).sort()[0].simplify()
        self.edge_hidden_irreps = (self.sh_irreps * self.edge_channels).sort()[0].simplify()
        self.hybrid_hidden_irreps = (self.sh_irreps * self.hybrid_channels).sort()[0].simplify()

        if self.branch_mode == "node":
            self.hidden_irreps = self.node_hidden_irreps
        elif self.branch_mode == "edge":
            self.hidden_irreps = self.edge_hidden_irreps
        else:
            self.hidden_irreps = self.hybrid_hidden_irreps
        self.output_dim = self.hidden_irreps.dim

        self.spherical_harmonics = SphericalHarmonics(
            self.sh_irreps,
            normalize=True,
            normalization="norm",
        )

        if self.num_radial_basis == 1:
            radial_centers = torch.tensor([0.5], dtype=torch.float32)
        else:
            radial_centers = torch.linspace(
                0.0,
                1.0,
                steps=self.num_radial_basis,
                dtype=torch.float32,
            )
        self.register_buffer("radial_centers", radial_centers)

        self.to(device)

    @staticmethod
    def _resolve_batch(data: Data) -> Tensor:
        batch = getattr(data, "batch", None)
        if batch is None:
            return torch.zeros(data.pos.size(0), dtype=torch.long, device=data.pos.device)
        return batch

    @staticmethod
    def _resolve_edge_index(data: Data) -> Tensor:
        edge_index = getattr(data, "edge_index", None)
        if edge_index is None:
            raise ValueError("ParaFreeModel requires data.edge_index.")
        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            raise ValueError(
                f"Expected edge_index with shape [2, E], got {tuple(edge_index.shape)}."
            )
        return edge_index

    def _resolve_base_node_attributes(self, data: Data) -> Tensor:
        x = getattr(data, "x", None)
        if x is None:
            x = getattr(data, "z", None)
        if x is None:
            raise ValueError("ParaFreeModel requires data.x or data.z to derive node attributes.")

        if x.dim() == 1:
            attrs = x.to(dtype=torch.float32).unsqueeze(-1)
        elif x.dim() == 2:
            attrs = x.to(dtype=torch.float32)
        else:
            raise ValueError(
                f"Expected node features with rank 1 or 2, got shape {tuple(x.shape)}."
            )

        if attrs.size(-1) == self.node_attr_dim:
            return attrs

        if self.node_attr_dim == 1:
            if attrs.size(-1) == 1:
                return attrs
            if attrs.size(-1) == 0:
                raise ValueError("Node feature width must be positive.")
            value_index = torch.arange(
                attrs.size(-1),
                device=attrs.device,
                dtype=attrs.dtype,
            )
            if torch.all((attrs >= -1e-6) & (attrs <= 1.0 + 1e-6)):
                row_sums = attrs.sum(dim=-1, keepdim=True)
                if torch.all(row_sums > 0):
                    return (attrs * value_index.unsqueeze(0)).sum(dim=-1, keepdim=True)
            return attrs.norm(dim=-1, keepdim=True)

        if attrs.size(-1) > self.node_attr_dim:
            raise ValueError(
                "Resolved node features are wider than node_input_dim. "
                f"Got width {attrs.size(-1)} with node_input_dim={self.node_attr_dim}."
            )

        pad = attrs.new_zeros(attrs.size(0), self.node_attr_dim - attrs.size(-1))
        return torch.cat([attrs, pad], dim=-1)

    def _resolve_node_attributes(self, data: Data) -> Tensor:
        attrs = self._resolve_base_node_attributes(data)
        override = getattr(data, "node_weights_override", None)
        if override is None:
            return attrs

        if override.dim() == 1:
            override = override.unsqueeze(-1)
        if override.dim() != 2:
            raise ValueError(
                "node_weights_override must have shape [N] or [N, C], "
                f"got {tuple(override.shape)}."
            )
        if override.size(0) != attrs.size(0):
            raise ValueError(
                "node_weights_override count does not match node count. "
                f"Got {override.size(0)} and {attrs.size(0)}."
            )

        override = override.to(dtype=torch.float32, device=attrs.device)
        if override.size(-1) == 1:
            return attrs * override
        if override.size(-1) == self.node_attr_dim:
            return override
        raise ValueError(
            "node_weights_override width must be 1 or node_input_dim. "
            f"Got width {override.size(-1)} with node_input_dim={self.node_attr_dim}."
        )

    def _resolve_edge_attributes(self, data: Data, pos: Tensor, edge_index: Tensor) -> Tensor:
        num_edges = int(edge_index.size(1))
        if self.edge_attr_dim == 0:
            return pos.new_zeros(num_edges, 0, dtype=torch.float32)

        edge_attr = getattr(data, "edge_attr", None)
        if edge_attr is None:
            row, col = edge_index
            distances = torch.linalg.norm(pos[col] - pos[row], dim=-1, keepdim=True)
            if self.edge_attr_dim == 1:
                return distances
            attrs = pos.new_zeros(num_edges, self.edge_attr_dim, dtype=torch.float32)
            attrs[:, :1] = distances
            return attrs

        if edge_attr.dim() == 1:
            attrs = edge_attr.to(dtype=torch.float32).unsqueeze(-1)
        elif edge_attr.dim() == 2:
            attrs = edge_attr.to(dtype=torch.float32)
        else:
            raise ValueError(
                f"Expected edge_attr with rank 1 or 2, got shape {tuple(edge_attr.shape)}."
            )

        if attrs.size(0) != num_edges:
            raise ValueError(
                "edge_attr count does not match edge_index. "
                f"Got {attrs.size(0)} and {num_edges}."
            )

        if attrs.size(-1) == self.edge_attr_dim:
            return attrs

        if self.edge_attr_dim == 1:
            if attrs.size(-1) == 1:
                return attrs
            return attrs.norm(dim=-1, keepdim=True)

        if attrs.size(-1) > self.edge_attr_dim:
            raise ValueError(
                "Resolved edge features are wider than edge_input_dim. "
                f"Got width {attrs.size(-1)} with edge_input_dim={self.edge_attr_dim}."
            )

        pad = attrs.new_zeros(attrs.size(0), self.edge_attr_dim - attrs.size(-1))
        return torch.cat([attrs, pad], dim=-1)

    def _compute_radial_basis(self, normalized_radius: Tensor) -> Tensor:
        diff = normalized_radius - self.radial_centers.view(1, -1)
        return torch.exp(-(diff.square()) / (2.0 * (self.radial_sigma ** 2)))

    def _graph_count(self, batch: Tensor) -> int:
        return int(batch.max().item()) + 1 if batch.numel() > 0 else 1

    def _build_node_blocks(
        self,
        pos: Tensor,
        batch: Tensor,
        node_attrs: Tensor,
    ) -> list[Tensor]:
        num_graphs = self._graph_count(batch)
        centers = scatter(pos, batch, dim=0, dim_size=num_graphs, reduce="mean")
        centered_pos = pos - centers[batch]

        radius = centered_pos.norm(dim=-1, keepdim=True)
        z_v = scatter(
            radius.squeeze(-1),
            batch,
            dim=0,
            dim_size=num_graphs,
            reduce="mean",
        ).unsqueeze(-1)
        radial_basis = self._compute_radial_basis(radius / (z_v[batch] + self.eps))
        directions = F.normalize(centered_pos, dim=-1, eps=self.eps)
        sh_values = self.spherical_harmonics(directions)
        # A node at the expansion centre has no direction. Its residual offset is pure
        # round-off, yet renormalised spherical harmonics would still return unit norm
        # at every degree, so mask everything above the invariant degree.
        at_center = radius <= self.center_tol * (z_v[batch] + self.eps)
        sh_values = torch.cat(
            [sh_values[:, :1], sh_values[:, 1:].masked_fill(at_center, 0.0)],
            dim=-1,
        )
        node_channels = (node_attrs.unsqueeze(-1) * radial_basis.unsqueeze(1)).reshape(
            pos.size(0),
            self.node_channels,
        )

        blocks: list[Tensor] = []
        sh_offset = 0
        for _, ir in self.sh_irreps:
            sh_block = sh_values[:, sh_offset : sh_offset + ir.dim]
            sh_offset += ir.dim
            node_block = node_channels.unsqueeze(-1) * sh_block.unsqueeze(1)
            graph_block = scatter(
                node_block,
                batch,
                dim=0,
                dim_size=num_graphs,
                reduce="sum",
            ).reshape(num_graphs, self.node_channels * ir.dim)
            blocks.append(graph_block)
        return blocks

    def _build_edge_blocks(
        self,
        pos: Tensor,
        batch: Tensor,
        node_attrs: Tensor,
        edge_index: Tensor,
        edge_attrs: Tensor,
    ) -> list[Tensor]:
        num_graphs = self._graph_count(batch)
        row, col = edge_index
        if row.numel() == 0:
            return [
                pos.new_zeros(num_graphs, self.edge_channels * ir.dim, dtype=torch.float32)
                for _, ir in self.sh_irreps
            ]

        edge_batch = batch[row]
        if not torch.equal(edge_batch, batch[col]):
            raise ValueError(
                "ParaFreeModel expects each edge to stay within a single graph batch element."
            )

        edge_vectors = pos[col] - pos[row]
        edge_radius = edge_vectors.norm(dim=-1, keepdim=True)
        z_e = scatter(
            edge_radius.squeeze(-1),
            edge_batch,
            dim=0,
            dim_size=num_graphs,
            reduce="mean",
        ).unsqueeze(-1)
        radial_basis = self._compute_radial_basis(edge_radius / (z_e[edge_batch] + self.eps))
        directions = F.normalize(edge_vectors, dim=-1, eps=self.eps)
        sh_values = self.spherical_harmonics(directions)

        edge_descriptor = torch.cat([node_attrs[row], node_attrs[col], edge_attrs], dim=-1)
        edge_channels = (edge_descriptor.unsqueeze(-1) * radial_basis.unsqueeze(1)).reshape(
            edge_descriptor.size(0),
            self.edge_channels,
        )

        blocks: list[Tensor] = []
        sh_offset = 0
        for _, ir in self.sh_irreps:
            sh_block = sh_values[:, sh_offset : sh_offset + ir.dim]
            sh_offset += ir.dim
            edge_block = edge_channels.unsqueeze(-1) * sh_block.unsqueeze(1)
            graph_block = scatter(
                edge_block,
                edge_batch,
                dim=0,
                dim_size=num_graphs,
                reduce="sum",
            ).reshape(num_graphs, self.edge_channels * ir.dim)
            blocks.append(graph_block)
        return blocks

    def _flatten_blocks(self, blocks: list[Tensor]) -> Tensor:
        return torch.cat(blocks, dim=-1)

    def _flatten_hybrid_blocks(
        self,
        node_blocks: list[Tensor],
        edge_blocks: list[Tensor],
    ) -> Tensor:
        hybrid_blocks: list[Tensor] = []
        for node_block, edge_block, (_, ir) in zip(node_blocks, edge_blocks, self.sh_irreps):
            node_view = node_block.view(node_block.size(0), self.node_channels, ir.dim)
            edge_view = edge_block.view(edge_block.size(0), self.edge_channels, ir.dim)
            hybrid_blocks.append(
                torch.cat([node_view, edge_view], dim=1).reshape(
                    node_block.size(0),
                    self.hybrid_channels * ir.dim,
                )
            )
        return self._flatten_blocks(hybrid_blocks)

    def _forward_default(
        self,
        pos: Tensor,
        batch: Tensor,
        node_attrs: Tensor,
        edge_index: Tensor,
        edge_attrs: Tensor,
    ) -> dict[str, Tensor]:
        node_blocks = self._build_node_blocks(pos=pos, batch=batch, node_attrs=node_attrs)
        edge_blocks = self._build_edge_blocks(
            pos=pos,
            batch=batch,
            node_attrs=node_attrs,
            edge_index=edge_index,
            edge_attrs=edge_attrs,
        )

        graph_sh_node = self._flatten_blocks(node_blocks)
        graph_sh_edge = self._flatten_blocks(edge_blocks)
        graph_sh_hybrid = self._flatten_hybrid_blocks(node_blocks, edge_blocks)

        if self.branch_mode == "node":
            graph_sh = graph_sh_node
        elif self.branch_mode == "edge":
            graph_sh = graph_sh_edge
        else:
            graph_sh = graph_sh_hybrid

        return {
            "graph_sh": graph_sh,
            "graph_sh_node": graph_sh_node,
            "graph_sh_edge": graph_sh_edge,
            "graph_sh_hybrid": graph_sh_hybrid,
        }

    def _encode_positions(
        self,
        pos: Tensor,
        node_weights: Tensor,
        *,
        center_positions: bool,
        radius_scale: Optional[Tensor | float] = None,
    ) -> Tensor:
        if pos.size(0) == 0:
            return torch.zeros(
                1,
                self.node_hidden_irreps.dim,
                dtype=torch.float32,
                device=node_weights.device,
            )

        work_pos = pos.to(dtype=torch.float32)
        work_weights = node_weights.to(dtype=torch.float32)
        if center_positions:
            work_pos = work_pos - work_pos.mean(dim=0, keepdim=True)

        radius = work_pos.norm(dim=-1, keepdim=True)
        if radius_scale is None:
            denom = radius.mean().clamp_min(self.eps)
        else:
            denom = torch.as_tensor(
                radius_scale,
                dtype=torch.float32,
                device=work_pos.device,
            ).reshape(())
            denom = denom.clamp_min(self.eps)

        radial_basis = self._compute_radial_basis(radius / denom)
        directions = F.normalize(work_pos, dim=-1, eps=self.eps)
        sh_values = self.spherical_harmonics(directions)
        weighted_rbf = work_weights * radial_basis

        graph_blocks = []
        sh_offset = 0
        for mul, ir in self.node_hidden_irreps:
            if int(mul) != self.node_channels:
                raise ValueError(
                    "Scalar component-mode path expects node_hidden_irreps multiplicity "
                    f"{self.node_channels}, got {mul} for irrep {ir}."
                )
            sh_block = sh_values[:, sh_offset : sh_offset + ir.dim]
            sh_offset += ir.dim
            node_block = weighted_rbf.unsqueeze(-1) * sh_block.unsqueeze(1)
            graph_blocks.append(node_block.sum(dim=0, keepdim=True).reshape(1, mul * ir.dim))

        return torch.cat(graph_blocks, dim=-1)

    def _encode_relative_vector(
        self,
        vector: Tensor,
        *,
        amplitude: Tensor | float,
        radius_scale: Tensor | float,
    ) -> Tensor:
        weight = torch.as_tensor(
            amplitude,
            dtype=torch.float32,
            device=vector.device,
        ).reshape(1, 1)
        if float(weight.item()) <= 0.0:
            return torch.zeros(
                1,
                self.node_hidden_irreps.dim,
                dtype=torch.float32,
                device=vector.device,
            )
        return self._encode_positions(
            vector.reshape(1, 3),
            weight,
            center_positions=False,
            radius_scale=radius_scale,
        )

    def _forward_component_mode(
        self,
        pos: Tensor,
        batch: Tensor,
        node_attrs: Tensor,
        component_role: Tensor,
    ) -> Tensor:
        if self.branch_mode != "node":
            raise ValueError(
                "component_mode is only supported with branch_mode='node' in ParaFreeModel."
            )
        if self.node_attr_dim != 1:
            raise ValueError(
                "component_mode currently requires node_input_dim=1 in ParaFreeModel."
            )

        num_graphs = self._graph_count(batch)
        rows = []

        for graph_index in range(num_graphs):
            graph_mask = batch == graph_index
            pos_graph = pos[graph_mask]
            attrs_graph = node_attrs[graph_mask]
            roles_graph = component_role[graph_mask]

            body_mask = roles_graph == 0
            water_mask = roles_graph == 1
            if not bool(body_mask.any()) or not bool(water_mask.any()):
                rows.append(
                    self._encode_positions(
                        pos_graph,
                        attrs_graph,
                        center_positions=True,
                    )
                )
                continue

            graph_sh = self._encode_positions(
                pos_graph[body_mask],
                attrs_graph[body_mask],
                center_positions=True,
            ) * self.body_local_scale
            graph_sh = graph_sh + (
                self._encode_positions(
                    pos_graph[water_mask],
                    attrs_graph[water_mask],
                    center_positions=True,
                )
                * self.water_local_scale
            )

            full_center = pos_graph.mean(dim=0, keepdim=True)
            full_extent = (pos_graph - full_center).norm(dim=-1).max()
            body_centroid = pos_graph[body_mask].mean(dim=0)
            water_centroid = pos_graph[water_mask].mean(dim=0)
            relative_vector = water_centroid - body_centroid
            relative_norm = relative_vector.norm()
            relative_scale = torch.maximum(full_extent, relative_norm)
            relative_weight = torch.sqrt(
                attrs_graph[body_mask].sum() * attrs_graph[water_mask].sum()
            ) * self.relative_vector_scale
            graph_sh = graph_sh + self._encode_relative_vector(
                relative_vector,
                amplitude=relative_weight,
                radius_scale=relative_scale,
            )
            rows.append(graph_sh)

        return torch.cat(rows, dim=0)

    def forward(self, data: Data):
        pos = data.pos.to(dtype=torch.float32)
        batch = self._resolve_batch(data)
        edge_index = self._resolve_edge_index(data)
        node_attrs = self._resolve_node_attributes(data).to(device=pos.device)
        edge_attrs = self._resolve_edge_attributes(data, pos, edge_index).to(device=pos.device)

        if pos.size(0) != node_attrs.size(0):
            raise ValueError(
                "Position and node-attribute counts do not match. "
                f"Got {pos.size(0)} and {node_attrs.size(0)}."
            )

        if self.component_mode != "none":
            component_role = getattr(data, "component_role", None)
            if component_role is None:
                raise ValueError(
                    "ParaFreeModel component_mode requires data.component_role to be present."
                )
            graph_sh = self._forward_component_mode(
                pos=pos,
                batch=batch,
                node_attrs=node_attrs,
                component_role=component_role.to(device=pos.device, dtype=torch.long),
            )
            outputs = {
                "graph_sh": graph_sh,
                "graph_sh_node": graph_sh,
                "graph_sh_edge": None,
                "graph_sh_hybrid": None,
            }
        else:
            outputs = self._forward_default(
                pos=pos,
                batch=batch,
                node_attrs=node_attrs,
                edge_index=edge_index,
                edge_attrs=edge_attrs,
            )
            graph_sh = outputs["graph_sh"]

        return build_model_output(
            graph_feat=graph_sh,
            graph_sh=graph_sh,
            node_pos=pos,
            graph_irreps=graph_sh,
            graph_sh_node=outputs["graph_sh_node"],
            graph_sh_edge=outputs["graph_sh_edge"],
            graph_sh_hybrid=outputs["graph_sh_hybrid"],
            branch_mode=self.branch_mode,
        )


ParameterFreeModel = ParaFreeModel


__all__ = ["ParaFreeModel", "ParameterFreeModel"]
