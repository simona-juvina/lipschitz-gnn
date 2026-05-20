# Lipschitz-GNN

Official implementation of the paper:

**Training Graph Neural Networks Subject to a Tight Lipschitz Constraint**  
Simona Ioana Juvina, Ana Antonia Neacsu, Jerome Rony, Jean-Christophe Pesquet, Corneliu Burileanu, Ismail Ben Ayed  
*Transactions on Machine Learning Research*, 2024

This repository contains code for training and evaluating the robustness of graph neural networks under tight Lipschitz constraints. It includes training scripts for baseline models, Lipschitz-constrained models, spectral-normalized models, adversarially trained models, randomized smoothing baselines, and graph-defense methods, together with attack and plotting utilities.

## Overview

This repository provides:

- training code for standard and Lipschitz-constrained graph neural networks;
- implementations of positivity constraints, spectral normalization, randomized smoothing, adversarial training, and graph-defense baselines;
- robustness evaluation under PGD and APGD attacks;
- plotting utilities for accuracy-robustness trade-off curves;
- scripts for reproducing the main experimental comparisons from the paper.

## Project Layout

```text
.
|-- scripts/                   # Runnable experiment entry points
|   `-- README.md              # Detailed script usage and experiment commands
|-- src/lipschitz_gnn/          # Reusable package code
|   |-- attacks/                # PGD/APGD attacks and attack metrics
|   |-- constraints.py          # Lipschitz, positivity, and spectral constraints
|   |-- early_stopping.py       # Early stopping callback
|   |-- models.py               # GNN model definitions
|   |-- plotting.py             # Plotting helpers
|   |-- training.py             # Training and evaluation loops
|   `-- utils.py                # Data loading, metrics, IO, Lipschitz helpers
|-- environment.yml             # Conda environment definition
|-- pyproject.toml              # Editable package and tooling configuration
`-- README.md
```

## Installation

On a Linux machine with at least one CUDA-enabled NVIDIA GPU and Anaconda or Miniconda installed, run:

```bash
git clone https://github.com/simona-juvina/lipschitz-gnn.git
cd lipschitz-gnn
conda env create --file environment.yml
conda activate robust_gnn
pip install -e .
```

Optional development tools can be installed with:

```bash
pip install -e ".[dev]"
black src scripts
ruff check src scripts
```

## Quick Start

The following example trains a Lipschitz-constrained GCN on `FacebookPagePage`, evaluates it with APGD, and plots the resulting robustness curve.

All commands should be run from the repository root after activating the conda environment.

```bash
# Train a Lipschitz-constrained GCN
python scripts/main_train_models.py \
  -db FacebookPagePage \
  -nt gcn \
  -wd 0.0005 \
  -ct full

# Evaluate robustness with APGD
python scripts/main_attack_models.py \
  -db FacebookPagePage \
  -nt gcn \
  -a apgd_l2_dl \
  -f results_attacks_apgd_l2_dl_lipschitz.csv

# Plot the accuracy-robustness trade-off
python scripts/main_plot_results.py \
  -cl 28 11 3 \
  -nl gcn \
  -f results_attacks_apgd_l2_dl_lipschitz.csv
```

For the full list of training, attack, graph-defense, randomized smoothing, adversarial training, and plotting commands, see [scripts/README.md](scripts/README.md).

## Supported Experiment Types

| Category | Supported methods |
|---|---|
| Standard training | Baseline GNN models |
| Lipschitz-constrained training | Tight Lipschitz-constrained models |
| Constraint variants | Full constraint, non-negative constraint, spectral normalization |
| Robust training baselines | Adversarial training, randomized smoothing |
| Graph-defense baselines | SVD, Jaccard, RGCN |
| Attacks | PGD and APGD variants under `L2` and `L-infinity` perturbations |
| Plotting | Accuracy and robustness trade-off curves |

## Main Scripts

| Script | Purpose |
|---|---|
| `scripts/main_train_models.py` | Train baseline, Lipschitz-constrained, non-negative, and spectral-normalized models |
| `scripts/main_attack_models.py` | Evaluate baseline and constrained models under PGD/APGD attacks |
| `scripts/main_train_models_RS.py` | Train randomized smoothing models |
| `scripts/main_attack_models_RS.py` | Evaluate randomized smoothing models |
| `scripts/main_train_models_AT.py` | Train adversarially trained models |
| `scripts/main_attack_models_AT.py` | Evaluate adversarially trained models |
| `scripts/main_train_attack_graph_defenses.py` | Train and evaluate graph-defense baselines |
| `scripts/main_plot_results.py` | Generate accuracy-robustness plots |

## Generated Outputs

The scripts create output directories automatically when needed.

| Output | Description |
|---|---|
| `saved_models/` | Trained model checkpoints |
| `saved_history/` | Training histories and intermediate metrics |
| `saved_figures/` | Generated plots |
| `results_attacks_*.csv` | Robustness evaluation results used by the plotting scripts |

## Citation

If you use this code, please cite our paper:

```bibtex
@article{juvina2024training,
  author  = {Juvina, Simona Ioana and Neac{\c{s}}u, Ana Antonia and Rony, J{\'e}r{\^o}me and Pesquet, Jean-Christophe and Burileanu, Corneliu and Ben Ayed, Ismail},
  title   = {Training Graph Neural Networks Subject to a Tight Lipschitz Constraint},
  journal = {Transactions on Machine Learning Research},
  year    = {2024},
  url     = {https://openreview.net/forum?id=KLojVqdj2y}
}
```
