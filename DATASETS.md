# Dataset definitions

All molecular coordinates are Cartesian coordinates in angstroms. Synthetic toy coordinates are dimensionless. `z` denotes atomic number; biological NPZ arrays retain atom ordering. Selection masks or indices refer to that exact order. Scores are computed after separately centring and representing each selection.

## Synthetic point groups and deformations

`toy/train.npz`, `val.npz` and `test.npz` retain 80/10/10 samples per point group. `ids`, `point_group`, `point_group_hm` and `group_order` have one entry per structure. `offsets` separates concatenated `pos` and `z` arrays. The test set is never used to select the shared rank tolerance: 3e-5 through degree six and 1e-6 through degree nine. The corresponding cached spectra and fixed-threshold results are in `source_data/fig2`.

The five `symmetry_breaking` families and four `mode_separation` families contain the actual constructed coordinates, XYZ trajectories and construction metadata. These are controlled synthetic experiments; the biological inputs below are deposited experimental structures. Analysis with natural-parity representations uses degrees 0–6 unless the dataset's protocol states otherwise.

## Biaryl torsion scans

Source: QCArchive, OpenFF Rowley Biaryl v1.0 (dataset 283). Record IDs are filenames in `torsion/trajectories`; `metadata.csv` and `responses.csv` associate coordinates, angles and deposited energies. `evaluated_pairs.json` lists the 94 target-conditioned evaluations over the 86 retained molecules. Each XYZ contains 24 optimized conformers on the 15-degree grid. Only the retained scans are included. Energies are reported in the units named in the CSV columns.

`source_data/fig4/engel_frames.csv` contains the orientation-optimized Engel scores for those 94 pairs (2,256 conformer–target evaluations). Blank target columns indicate pairs outside the evaluated cohort. The descriptor and search settings are recorded in `source_data/engel_protocol.json`.

## LnDOTA

Source studies: Manvell et al., Chemical Science 15, 113–123 (2024), and the Nielsen/Sørensen CSoM data release. CCDC entries 2281329 and 2281615–2281620 give the seven water-free crystallographic clusters. `metadata.csv` records the source entry of each derived cluster; the deposited space-group expansion is already included in the XYZ coordinates. Two additional approximate-symmetry coordination clusters are supplied with their source DOI. Hydrogen presence follows the per-cluster metadata; use the atom selection specified by the experiment when recomputing a score.

The `water_rotation` directory contains the 360 original CSoM reference XYZ files. The upstream GPL-3.0 text is retained as `UPSTREAM_LICENSE`; it applies to the upstream material and does not relicense unrelated project files.

The four `engel_*` columns in `source_data/fig5/extra_baselines_frames.csv` use the same enhanced shared-orientation search as Fig. 4, applied independently to each frame and target. The two `selfmatch_*` columns retain their existing coordinate scores.

## p97

Sources: PDB 8FCL, 8FCM, 8FCR, 8FCQ, 8FCO and 8FCT; Braxton et al. (2023). `deposited/` contains the source mmCIF files. `matched_coordinates.npz` contains 17,088 N/CA/C/O atoms per structure, in the same chain/atom order across all six states. `resid`, `atom`, `z` and `chain` are shared arrays; each PDB identifier names its coordinate array. Numeric chain indices map to the ordered chain names in `protocol.json`.

`regions.json` defines the whole selection, D1, D2 and the 26 windows used in Fig. 6. Bounds are inclusive residue numbers. Extended Data Fig. 2 uses widths 20/40/60/80: `source_data/exfig2/window_scores.csv` gives each actual start/end, nominal extent, clipped-boundary flag, atom count and reference difference, including all six structures (672 scores). `neighbour_scores.csv` contains the 70 plotted differences for the five non-reference structures. The reference is 8FCL. p97 graphs include the tied kth-distance shell; the supplied scorer uses the original float32 torch-distance convention.

## Haemagglutinin

Sources: PDB 6Y5G–6Y5K; Benton et al. (2020). Each NPZ stores coordinates in the deposited atom order and Boolean `whole`, `HA1`, `HA2` masks. HA1 uses author residues 45–310 on chains A/C/E; HA2 uses residues 38–125 on B/D/F. The global selection is their union, not every atom in the mmCIF file. Unresolved residues are not filled. State III has fewer atoms because of deposited coverage. HA uses float64 centring followed by float32 conversion and fixed k=24 neighbours. `source_data/exfig3/spectra.json` provides complete spectra for the 15 state/region selections.

## GluK2

Source: PDB 5KUF biological assembly 1; Meyerson et al. (2016). `common_coordinates.npz` contains the 12,000-atom common complete backbone, with deposited identifiers. `domains.json` gives the three domain selections. `layers.json` specifies the TMD-to-ATD axis, origin, slab limits and exact atom indices of all 44 layers over thicknesses 10/20/30/40/50/60 angstroms. Slabs are non-overlapping within each thickness; the last remainder is evaluated separately. The axis origin differs from the display origin at the bottom of the structure; use the explicit indices to reproduce the selections. The displayed score is pLS for C4. `source_data/exfig4/spectra.json` contains the archived spectra for the three domains and 44 layers.

## Echovirus 18

Sources: biological assembly 1 of PDB 6HBH, 6HBK, 6HHT and 6HBL; Buchta et al. (2019). Source compressed mmCIF files and matched C-alpha coordinates are both supplied. These are deposited structures; missing pentamers were not removed computationally. `sequence_alignment.json` documents the mapping from deposited numbering to common capsid-protein positions. The deposited author numbering is retained separately as `resid_author`; missing physical chains remain absent. The analysis targets the 60 proper icosahedral rotations and reports the degree-six pLS with a target fixed-subspace dimension of one. The supplied values are the saved degree-six results, not pooled degree 0–6 values.

## Computational measurements

`source_data/computational_cost/trial_source_data.csv` preserves trial roles, completion status and times. Use the published repetition schedule to compute medians: warm-up/calibration rows are not measured trials. In these earlier records, orientation-fitting configurations have one timed trial; a censored result is not an observed completion time. `protocol.json` records timing boundaries, precision, threads and the actual hardware. The data include supporting repeated measurements; no hardware conversion is applied.

Engel retained-energy search timings are in `engel_search_trials.csv` and `engel_search_summary.csv` in the same directory, with settings in `engel_search_protocol.json`. They cover grid search, BFGS, and BFGS with a CSoM-fitted start. Current Engel timings use three repetitions for all DOTA searches and for biological grid/BFGS searches without a CSoM-fitted start; biological searches with a CSoM-fitted start use one trial. Completed repeated runs are summarized by their median. The p97 timeout is a separately measured 7,200-second per-call bound; it has no completed runtime or final score. In the earlier `summary.csv` and `trial_source_data.csv`, Engel rows with `orientation=search` measure coordinate fitting alone, not retained-energy maximization.

The current SymFP timing inputs are in `datasets/computational_cost/`; runnable commands and measurement records are described in `source_data/computational_cost/README.md` after extraction.

## Torsion pretraining

Both pretraining objectives use `datasets/torsion/` directly. `code/scripts/prepare_torsion_data.py` generates the original seed-42 69/17 molecule split and train-only energy normalization in the local output directory. Coordinates and energy labels are stored only once in the main dataset. `point_groups.csv` contains the original per-conformation point-group assignments at tolerances 0.1 and 0.3; `evaluated_pairs.json` defines the 94 molecule–target pairs.

`configs/torsion_pretraining.json` records checkpoint selection and file hashes. `source_data/torsion_pretraining/scores/` contains cached pLS values for all 12 model/variant combinations. These geometries inherit the QCArchive OpenFF Rowley Biaryl v1.0 provenance above. See [the pretraining guide](docs/torsion_pretraining.rst) for commands and definitions.

Datasets and numerical source data are distributed in `datasets.zip` and `source_data.zip`; `archives.json` records extracted-file checksums. `demo/data/` retains independent copies of its example inputs so the notebooks can run without extracting the archives.
