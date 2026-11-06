# Sources and third-party material

The input export retains source accessions, citations and the applicable upstream notice for the CSoM reference data. Scientific provenance is also recorded in DATASETS.md and the per-dataset metadata.

- Protein Data Bank experimental coordinates: accessions listed in DATASETS.md; source mmCIF bibliographic/author metadata is preserved.
- QCArchive OpenFF Rowley Biaryl v1.0: retained optimized coordinates and energies with record IDs.
- LnDOTA: derived coordination clusters from the cited CCDC entries and reference coordinates from the CSoM data release. The upstream GPL-3.0 notice accompanies the latter.
- Learned extractor implementations: TFN (Thomas et al., 2018), MACE (Batatia et al., 2022), Equiformer (Liao and Smidt, 2023), and HEGNN (Cen et al., 2024), with the manuscript's multi-channel HEGNN extension. Source code comments and attribution are retained. The upstream MACE and Equiformer MIT notices are included in `code/third_party_licenses/` for their incorporated code; SymFP-owned code is separately covered by the project PolyForm Noncommercial License 1.0.0.

Dependencies are installed separately and retain their own licences. The project PolyForm Noncommercial License 1.0.0 applies to SymFP-owned code, accompanying documentation and supplied model weights. It does not replace licences for third-party materials or relicense the original datasets.

Additional retained model baselines are EGNN, SchNet, DimeNet++ and GVP, implemented in `code/models/`. PyTorch Geometric and separately installed baseline programs retain their own licences. The bundled CSoM implementation and wrappers are distributed here; third-party standalone executables are not bundled.

The torsion denoising objective adapts the masked-atom, coordinate and pair-distance recovery tasks of Uni-Mol (Zhou et al., 2023). Protocol reference: `deepmodeling/Uni-Mol`, commit `90f52c41299a1a582da0f9765e9f87aa21faa16a`. The task heads and data corruption here are implemented for the four bundled equivariant GNNs; this is not the original Uni-Mol architecture or its large-scale pretraining dataset.
