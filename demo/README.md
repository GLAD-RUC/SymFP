# Interactive SymFP demos

Explore how structural changes appear in SymFP spectra and symmetry scores.
Start with the live notebook to try your own coordinates, or use the three
lightweight notebooks to explore bundled deformation results.

| Notebook | Experiment / analysis | What you can change |
| --- | --- | --- |
| [custom_structure.ipynb](custom_structure.ipynb) | Live SymFP calculation and structural deformations | Example or single-frame XYZ input; strain, twist and noise |
| [rank_activation.ipynb](rank_activation.ipynb) | Rank activation under symmetry breaking | Three prescribed deformation paths |
| [subgroup_chain.ipynb](subgroup_chain.ipynb) | Leakage along a subgroup chain | Five target point groups |
| [angular_modes.ipynb](angular_modes.ipynb) | Spectral onset for angular and axial modes | Four displacement fields |

## Setup

All four notebooks run on the CPU using NumPy, SciPy, pandas and Matplotlib, with
JupyterLab for the interface. They do not require PyTorch, e3nn, PyG, pymatgen,
CUDA, model weights or the training environment. With Python 3.10, run from the
repository root:

```bash
python -m venv .venv-demo
source .venv-demo/bin/activate
python -m pip install -r demo/requirements.txt
python -m ipykernel install --user --name symfp-demo --display-name "SymFP demo"
python -m jupyterlab demo/
```

On Windows, activate the environment with `.venv-demo\Scripts\activate` instead.
Select **SymFP demo** and choose **Restart Kernel and Run All Cells**. Installation
needs network access; subsequent notebook execution uses local files.

## Live calculation

`custom_structure.ipynb` computes the parameter-free hybrid representation directly
from coordinates. It shows the original and modified structures, singular-value
spectra, numerical ranks, and LS/pLS for 32 crystallographic target groups.
The NumPy/SciPy implementation preserves the PF radial weights, node and edge
channels, spherical-harmonic normalization, and degree-wise leakage pooling. It
uses float64; near-zero feature blocks can therefore give different leakage
values from the float32 PyTorch implementation. The 32 rank templates are fixed-subspace dimensions;
features and scores are computed from the supplied coordinates at runtime.

Open `custom_structure.ipynb` and run all cells for the default bundled example.
Edit `STRETCH`, `TWIST` or `NOISE` and run the cells again to compare deformations.

To use your own structure, place a single-frame XYZ file in `demo/` through Jupyter's
file browser, set `SOURCE = "xyz"` and `XYZ_PATH = "demo/your_structure.xyz"`, then
run all cells. Paths are relative to the repository root; absolute paths also work.
Set all three deformation parameters to zero to inspect the unmodified input.

The live demo accepts finite, nonperiodic structures with 2–128 atoms and standard
element symbols. It uses all directed atom pairs, so work grows quadratically with
atom count. The notebook displays the results without exporting them. LS/pLS report
agreement with target rank constraints; they do not assign a unique point group.

## Bundled-result notebooks

`rank_activation.ipynb`, `subgroup_chain.ipynb` and `angular_modes.ipynb` read bundled
structures and precomputed SymFP results. They use the same **SymFP demo** kernel.
Change `CASE`, `TARGET` or `MODE` in the configuration cell to explore another
selection, then run all cells again.

## Bundled data

`data/` contains approximately 280 KB of synthetic structures, metadata and cached
results. `data/manifest.json` records the data paths, sizes and SHA-256 checksums;
the repository's `files.json` also covers the complete demo. No pickle files are
loaded by these notebooks.

The cached notebooks check rank saturation and spectral onset for their selected
examples. Fixed-subspace dimensions are rank upper bounds; the angular-onset
interpretation concerns the four prescribed fields. Use the live notebook for new
coordinates and exploratory deformations.
