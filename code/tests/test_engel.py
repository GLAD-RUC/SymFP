"""Numerical checks of symmetry identities, gradients and orientation search."""

import json
from pathlib import Path
import sys
import unittest

import numpy as np
import torch
from e3nn import o3
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

from baseline.csom_modules.core import LabeledPointCloud
from baseline.csom_modules.groups import canonical_label_pair
from models.symmetry_groups import invariant_multiplicity
from baseline.engel import EngelGap, _build_harmonic_descriptor, _rotate_descriptor
from baseline.engel_orientation import ProjectionObjective, projector, target_operations


def read_xyz(path, frame=0):
    with path.open() as stream:
        for _ in range(frame + 1):
            count = int(stream.readline())
            stream.readline()
            rows = [stream.readline().split() for _ in range(count)]
    return LabeledPointCloud(path.stem, tuple(r[0] for r in rows),
                             np.array([[float(v) for v in r[1:4]] for r in rows]))


class EngelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_projectors_match_fixed_subspace_dimensions(self):
        groups = json.loads((ROOT / "code/baseline/engel_point_groups.json").read_text())
        self.assertEqual(len(groups), 32)
        for label in groups:
            _, hm = canonical_label_pair(label)
            self.assertIsNotNone(hm)
            operations = target_operations(label)
            for a in operations:
                np.testing.assert_allclose(a @ a.T, np.eye(3), atol=1e-12)
                for b in operations:
                    self.assertLess(min(np.max(abs(a @ b - c)) for c in operations), 1e-12)
            for ell in range(7):
                p = projector(label, ell)
                np.testing.assert_allclose(p @ p, p, atol=1e-12)
                np.testing.assert_allclose(p.T, p, atol=1e-12)
                expected = invariant_multiplicity(hm, ell, "e" if ell % 2 == 0 else "o")
                self.assertEqual(round(np.trace(p)), expected)

    def test_analytic_gradient(self):
        rng = np.random.default_rng(23)
        objective = ProjectionObjective(tuple(rng.normal(size=2*l+1) for l in range(7)), "C4v")
        angles = np.array([0.37, 0.82, -0.54])
        _, gradient = objective.value_gradient(angles)
        eps = 1e-6
        numeric = np.array([
            (objective.value_gradient(angles + eps * d)[0]
             - objective.value_gradient(angles - eps * d)[0]) / (2 * eps)
            for d in np.eye(3)
        ])
        np.testing.assert_allclose(gradient, numeric, atol=2e-9, rtol=1e-6)

    def test_representation_rotation_matches_rotated_coordinates(self):
        points = np.random.default_rng(19).normal(size=(7, 3))
        rotation = Rotation.from_rotvec([0.32, -0.47, 0.13]).as_matrix()
        config = dict(max_ell=6, harmonic_normalization="integral",
                      descriptor_mode="pair_directions", zero_radius_eps=1e-12)
        original = _build_harmonic_descriptor(points, **config)
        rotated = _build_harmonic_descriptor(points @ rotation.T, **config)
        for actual, expected in zip(_rotate_descriptor(original, rotation), rotated.blocks):
            np.testing.assert_allclose(actual, expected, atol=2e-13)

    def test_streamed_pairs_match_direct_harmonic_mean(self):
        points = np.random.default_rng(73).normal(size=(270, 3))
        descriptor = _build_harmonic_descriptor(
            points, max_ell=6, harmonic_normalization="integral",
            descriptor_mode="pair_directions", zero_radius_eps=1e-12,
        )
        raw = (points[None, :, :] - points[:, None, :]).reshape(-1, 3)
        raw = raw[np.linalg.norm(raw, axis=1) > 1e-12]
        direct = o3.spherical_harmonics(list(range(7)), torch.as_tensor(raw),
                                        normalize=True, normalization="integral").mean(0).numpy()
        np.testing.assert_allclose(np.concatenate(descriptor.blocks), direct, atol=2e-13)
        self.assertEqual(descriptor.num_directions_used, 270 * 269)

    def test_rotated_mirror_plane_and_orientation_free_groups(self):
        points = np.array([[0.1, 0.3, 0.], [1.2, -0.2, 0.], [-0.3, 1.4, 0.], [-0.8, -0.6, 0.]])
        rotation = Rotation.from_rotvec([0.7, -0.6, 0.4]).as_matrix()
        sample = LabeledPointCloud("mirror", ("C",) * len(points), points @ rotation.T)
        given = EngelGap(candidate_groups=["Cs"], axis_mode="none")(sample).predicted
        fitted = EngelGap(candidate_groups=["Cs"])(sample).predicted
        without_coordinate_start = EngelGap(candidate_groups=["Cs"], use_csom_start=False)(sample).predicted
        self.assertLess(given.engel_score, 0.99)
        self.assertAlmostEqual(fitted.engel_score, 1., places=10)
        self.assertLess(fitted.engel_gap, 1e-10)
        self.assertGreaterEqual(without_coordinate_start.engel_score + 1e-12, given.engel_score)
        self.assertFalse(any(r["stage"] == "coordinate_start" for r in without_coordinate_start.search_diagnostics))
        for label in ("C1", "Ci"):
            result = EngelGap(candidate_groups=[label])(sample).predicted
            self.assertAlmostEqual(result.engel_score, 1., places=12)
            self.assertEqual(result.search_diagnostics, ())

    def test_default_search_beats_given_and_coordinate_fitted_frames(self):
        sample = read_xyz(ROOT / "datasets/torsion/trajectories/21272352.xyz")
        given = EngelGap(candidate_groups=["Cs"], axis_mode="none")(sample).predicted
        coordinate = EngelGap(candidate_groups=["Cs"], axis_mode="csom_search")(sample).predicted
        fitted = EngelGap(candidate_groups=["Cs"])(sample).predicted
        self.assertEqual(fitted.orientation_mode, "engel_search")
        self.assertGreater(fitted.engel_score, coordinate.engel_score + 0.01)
        self.assertGreaterEqual(fitted.engel_score + 1e-12, given.engel_score)
        self.assertTrue(any(r["stage"] == "coordinate_start" for r in fitted.search_diagnostics))
        self.assertAlmostEqual(fitted.engel_score, (fitted.alpha - fitted.beta) / (1 - fitted.beta), places=12)



if __name__ == "__main__":
    unittest.main()
