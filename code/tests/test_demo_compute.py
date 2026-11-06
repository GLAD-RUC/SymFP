"""CPU demo checks; reference comparisons additionally use the scientific environment."""
from importlib.util import find_spec
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.special import eval_legendre

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from utils.demo_compute import (
    LiveSymFP, _ATOMIC_NUMBERS, _real_spherical_harmonics,
    deform_structure, read_xyz, validate_structure,
)


class CPUDemoTests(unittest.TestCase):
    def test_harmonic_normalization_and_addition_theorem(self):
        vectors = np.random.default_rng(17).normal(size=(18, 3))
        unit = vectors / np.linalg.norm(vectors, axis=1)[:, None]
        for ell in range(7):
            harmonics = _real_spherical_harmonics(vectors, ell)
            assert_allclose(harmonics @ harmonics.T,
                            eval_legendre(ell, np.clip(unit @ unit.T, -1, 1)), atol=2e-14)

    def test_rigid_motion_and_atom_order(self):
        rng = np.random.default_rng(9)
        positions = rng.normal(size=(12, 3))
        symbols = np.array(["C", "H", "N", "O"] * 3)
        calculator = LiveSymFP()
        reference = calculator.compute(symbols, positions)
        rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        permutation = rng.permutation(len(positions))
        # Include an improper orthogonal transformation as well as a translation.
        changed = calculator.compute(symbols[permutation],
                                     (-positions @ rotation + [2, -4, 1])[permutation])
        for a, b in zip(reference["spectra"], changed["spectra"]):
            assert_allclose(a, b, rtol=1e-11, atol=2e-12)
        assert_array_equal(reference["ranks"], changed["ranks"])
        assert_allclose(reference["scores"].iloc[:, 1:], changed["scores"].iloc[:, 1:], atol=1e-10)

    def test_two_atom_analytic_spectra_and_channel_capacity(self):
        for channels in (1, 16, 32):
            calculator = LiveSymFP(channels=channels)
            result = calculator.compute(["C", "C"], [[-1, 0, 0], [1, 0, 0]])
            # Even degrees have one nonzero singular value. The four channel
            # families have amplitudes 12, 12, 12, 4 times the same radial basis.
            radial = np.exp(-0.5 * ((1 / (1 + 1e-10) - calculator.radial_centers) * channels) ** 2)
            edge_radial = np.exp(-0.5 * ((2 / (2 + 1e-10) - calculator.radial_centers) * channels) ** 2)
            expected = np.sqrt(144 * np.dot(radial, radial) + 304 * np.dot(edge_radial, edge_radial))
            for ell, values in enumerate(result["spectra"]):
                target = np.zeros(min(4 * channels, 2 * ell + 1))
                if ell % 2 == 0:
                    target[0] = expected
                assert_allclose(values, target, atol=2e-13)
            assert_array_equal(result["bounds"].loc["1"],
                               np.minimum(np.arange(1, 14, 2), 4 * channels))
            scores = result["scores"].set_index("candidate (Hermann–Mauguin)")
            self.assertEqual(scores.loc["1", "LS (lower = more compatible)"], 0)
            self.assertEqual(scores.loc["1", "pLS"], 10)

    def test_octahedral_ranks_center_atom_and_deformation(self):
        positions = np.concatenate((np.eye(3), -np.eye(3), np.zeros((1, 3))))
        symbols = ["O"] * 6 + ["Dy"]
        calculator = LiveSymFP()
        result = calculator.compute(symbols, positions)
        assert_array_equal(result["ranks"], [1, 0, 0, 0, 1, 0, 1])
        scores = result["scores"].set_index("candidate (Hermann–Mauguin)")
        self.assertLess(scores.loc["m-3m", "LS (lower = more compatible)"], 1e-14)
        changed = calculator.compute(symbols, deform_structure(positions, stretch=0.15))
        self.assertGreater(changed["ranks"][2], result["ranks"][2])
        changed_scores = changed["scores"].set_index("candidate (Hermann–Mauguin)")
        self.assertGreater(changed_scores.loc["m-3m", "LS (lower = more compatible)"], 0.01)

    def test_xyz_and_input_validation(self):
        self.assertEqual(len(_ATOMIC_NUMBERS), 118)
        self.assertEqual(_ATOMIC_NUMBERS["Dy"], 66)
        self.assertEqual(_ATOMIC_NUMBERS["Og"], 118)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.xyz"
            path.write_text("2\nexample\nC 1 0 0\nH 3 0 0\n")
            symbols, positions = read_xyz(path)
            self.assertEqual(symbols, ["C", "H"])
            assert_array_equal(positions, [[-1, 0, 0], [1, 0, 0]])
            path.write_text(path.read_text() * 2)
            with self.assertRaisesRegex(ValueError, "one XYZ frame"):
                read_xyz(path)
        invalid = [(["C", "Xx"], [[0, 0, 0], [1, 0, 0]]),
                   (["C", "H"], [[0, 0, 0], [0, 0, 0]]),
                   (["C", "H"], [[0, 0, 0], [np.nan, 0, 0]])]
        for symbols, positions in invalid:
            with self.subTest(symbols=symbols, positions=positions), self.assertRaises(ValueError):
                validate_structure(symbols, positions)


HAS_REFERENCE = all(find_spec(module) is not None for module in
                    ("torch", "e3nn", "torch_geometric", "torch_scatter", "pymatgen"))


@unittest.skipUnless(HAS_REFERENCE, "Optional comparison requires the scientific environment")
class ReferenceTests(unittest.TestCase):
    def test_all_rank_templates(self):
        from models.symmetry_groups import POINT_GROUPS, invariant_multiplicity
        calculator = LiveSymFP()
        self.assertEqual(calculator.point_groups, POINT_GROUPS)
        expected = [[invariant_multiplicity(group, ell, "e" if ell % 2 == 0 else "o")
                     for ell in range(7)] for group in POINT_GROUPS]
        assert_array_equal(calculator.observable_rank_matrix, expected)

    def test_pf_spectra_against_original_feature_builders(self):
        import torch
        from models.para_free import ParaFreeModel
        from models.symfp_wrapper import SymFPWrapper
        torch.set_num_threads(1)
        rng = np.random.default_rng(13)
        cases = [(np.array([6, 8, 1, 66, 7, 8, 6, 1]), rng.normal(size=(8, 3))),
                 (np.array([6, 6]), np.array([[-1., 0, 0], [1., 0, 0]])),
                 (np.array([8] * 6 + [66]),
                  np.concatenate((np.eye(3), -np.eye(3), np.zeros((1, 3)))))]
        for channels in (1, 16, 32):
            encoder = ParaFreeModel(max_ell=6, irreps_channels=channels, branch_mode="hybrid").double()
            # Construct the same analytic radial centres at double precision to
            # compare the formulas independently of float32 parameter rounding.
            encoder.radial_centers = (
                torch.tensor([0.5], dtype=torch.float64) if channels == 1
                else torch.linspace(0, 1, channels, dtype=torch.float64)
            )
            wrapper = SymFPWrapper(encoder)
            calculator = LiveSymFP(channels=channels)
            elements = {number: symbol for symbol, number in _ATOMIC_NUMBERS.items()}
            for atomic, positions in cases:
                centered = positions - positions.mean(axis=0)
                pos = torch.from_numpy(centered)
                attrs = torch.tensor(atomic[:, None], dtype=torch.float64)
                batch = torch.zeros(len(pos), dtype=torch.long)
                edges = torch.ones((len(pos), len(pos)), dtype=torch.bool).fill_diagonal_(False).nonzero().T
                distances = torch.linalg.vector_norm(pos[edges[1]] - pos[edges[0]], dim=-1, keepdim=True)
                with torch.inference_mode():
                    node = encoder._build_node_blocks(pos, batch, attrs)
                    edge = encoder._build_edge_blocks(pos, batch, attrs, edges, distances)
                actual = calculator.compute([elements[z] for z in atomic], centered)
                expected_spectra = []
                for ell, (n, e, spectrum) in enumerate(zip(node, edge, actual["spectra"])):
                    reference = torch.cat((n.reshape(channels, 2 * ell + 1),
                                           e.reshape(3 * channels, 2 * ell + 1)), dim=0)
                    expected_tensor = torch.linalg.svdvals(reference)
                    expected_spectra.append(expected_tensor[None, :])
                    expected = expected_tensor.numpy()
                    scale = max(expected[0], 1)
                    assert_allclose(spectrum / scale, expected / scale, rtol=1e-11, atol=1e-12)
                expected_ls, expected_pls = wrapper._compute_soft_scores(expected_spectra)
                assert_allclose(actual["scores"].iloc[:, 1], expected_ls[0].numpy(), atol=1e-11)
                assert_allclose(actual["scores"].iloc[:, 2], expected_pls[0].numpy(), atol=1e-8)


if __name__ == "__main__":
    unittest.main()
