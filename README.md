# Spectral symmetry fingerprints in equivariant representations

[![arXiv](https://img.shields.io/badge/arXiv-2611.07410-b31b1b.svg)](https://arxiv.org/abs/2611.07410)

SymFP uses singular-value spectra of equivariant representations for discrete point-group identification and continuous symmetry analysis. This repository provides the implementation, datasets, model weights, numerical source data and four interactive demos.

## Quick start

```bash
git clone https://github.com/GLAD-RUC/SymFP.git
cd SymFP
python -m venv .venv-demo
source .venv-demo/bin/activate
python -m pip install -r demo/requirements.txt
python -m ipykernel install --user --name symfp-demo --display-name "SymFP demo"
python -m jupyterlab demo/
```

Select the **SymFP demo** kernel. Start with [custom_structure.ipynb](demo/custom_structure.ipynb) to analyze an example structure or your own XYZ file. The other notebooks demonstrate rank activation, subgroup relations and deformation modes. All four run on the CPU; see the [demo guide](demo/README.md).

## Data and models

Extract the archives from the repository root:

```bash
python -m zipfile -e datasets.zip .
python -m zipfile -e checkpoints.zip .
python -m zipfile -e source_data.zip .
```

| Path | Contents |
|---|---|
| `datasets/` | Synthetic point clouds, molecular and biological coordinates, labels and dataset splits |
| `checkpoints/` | 8 point-group benchmark models and 12 torsion models |
| `configs/` | Model and training settings |
| `source_data/` | Scores, spectra, input-setting checks and computational benchmark records |
| `demo/` | Interactive notebooks and their supporting data |

[DATASETS.md](DATASETS.md) describes the inputs and source data. To verify file checksums and inspect the datasets, run `python code/inspect_data.py --verify` with NumPy installed.

## Analysis and training

Install the scientific environment:

```bash
conda env create -f code/environment.yml
conda activate SymFP
```

Run from the repository root:

```bash
# Analyze a molecular structure.
python code/scripts/analyze_xyz.py datasets/lndota/clusters/Dy_DOTA_H2O.xyz --target C4

# Evaluate point-group identification.
python code/scripts/evaluate_toy.py --model para_free

# Recompute a biological selection.
python code/recompute_pf.py p97 --structure 8FCL --region D1

# Train an equivariant GNN.
python code/scripts/toy_supervised.py --model hegnn --logger none --test_noise_levels 0

# Analyze the supplied torsion scores.
python code/scripts/torsion_pretraining.py analyze --scores source_data/torsion_pretraining/scores --out outputs/torsion_tables
```

Use `--help` for command options. The [torsion pretraining guide](docs/torsion_pretraining.rst) covers training, score extraction and analysis. Run `python -m unittest discover -s code/tests` for numerical checks.

## Citation

If you find SymFP useful in your research, please cite our [paper](https://arxiv.org/abs/2611.07410):

```bibtex
@article{cen2026spectral,
  title={Spectral symmetry fingerprints in equivariant representations},
  author={Cen, Jiacheng and Li, Anyi and Lin, Ning and Wang, Zihe and Wen, Ji-Rong and Huang, Wenbing},
  journal={arXiv preprint arXiv:2611.07410},
  year={2026}
}
```

## License

SymFP code, documentation and model weights use the [PolyForm Noncommercial License 1.0.0](LICENSE). Dataset and third-party terms are listed in [DATASETS.md](DATASETS.md) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
