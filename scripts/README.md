# Experiment Scripts

This folder contains the main entry-point scripts used to train graph neural networks, evaluate robustness, run baseline methods, and generate plots for:

**Training Graph Neural Networks Subject to a Tight Lipschitz Constraint**  
Simona Ioana Juvina, Ana Antonia Neacsu, Jerome Rony, Jean-Christophe Pesquet, Corneliu Burileanu, Ismail Ben Ayed  
*Transactions on Machine Learning Research*, 2024

All commands in this file should be run from the repository root after activating the conda environment:

```bash
conda activate robust_gnn
```

For installation instructions and citation information, see the root [README.md](../README.md).

## Script Overview

| Script | Purpose |
|---|---|
| `main_train_models.py` | Train baseline, Lipschitz-constrained, non-negative, and spectral-normalized models |
| `main_attack_models.py` | Evaluate baseline and constrained models under PGD/APGD attacks |
| `main_train_models_RS.py` | Train randomized smoothing models |
| `main_attack_models_RS.py` | Evaluate randomized smoothing models |
| `main_train_models_AT.py` | Train adversarially trained models |
| `main_attack_models_AT.py` | Evaluate adversarially trained models |
| `main_train_attack_graph_defenses.py` | Train and evaluate graph-defense baselines such as SVD, Jaccard, and RGCN |
| `main_plot_results.py` | Generate accuracy-robustness plots |

## Common Arguments

| Argument | Description | Example |
|---|---|---|
| `-db` | Dataset name | `FacebookPagePage` |
| `-nt` | Network, model, or defense type | `gcn`, `svd`, `jaccard`, `rgcn` |
| `-wd` | Weight decay | `0.0005` |
| `-ct` | Constraint type | `full`, `positive`, `spectral` |
| `-a` | Attack type | `apgd_l2_dl`, `apgd_linf`, `pgd_l2_ce` |
| `-f` | Output CSV file | `results_attacks_apgd_l2_dl_lipschitz.csv` |
| `-el` | Evaluation perturbation radii | `0.01 0.05 0.1 0.2` |
| `-cl` | Constraint/Lipschitz parameter values used for plotting | `28 11 3` |
| `-nl` | Network label used for plotting | `gcn` |
| `-fa` | Adversarial training results file used for plotting | `results_attacks_apgd_l2_dl_AT.csv` |
| `-frs` | Randomized smoothing results file used for plotting | `results_attacks_apgd_l2_dl_RS.csv` |
| `-fsn` | Spectral normalization results file used for plotting | `results_attacks_apgd_l2_dl_SN.csv` |
| `-fj` | Jaccard defense results file used for plotting | `results_attacks_apgd_l2_dl_jaccard.csv` |
| `-fr` | RGCN results file used for plotting | `results_attacks_apgd_l2_dl_rgcn.csv` |
| `-fsvd` | SVD defense results file used for plotting | `results_attacks_apgd_l2_dl_svd.csv` |

The exact available options are defined in the corresponding scripts.

## Quick Pipeline

The following commands train a Lipschitz-constrained GCN, evaluate it with APGD, and plot the resulting accuracy-robustness curve.

```bash
python scripts/main_train_models.py \
  -db FacebookPagePage \
  -nt gcn \
  -wd 0.0005 \
  -ct full

python scripts/main_attack_models.py \
  -db FacebookPagePage \
  -nt gcn \
  -a apgd_l2_dl \
  -f results_attacks_apgd_l2_dl_lipschitz.csv

python scripts/main_plot_results.py \
  -cl 28 11 3 \
  -nl gcn \
  -f results_attacks_apgd_l2_dl_lipschitz.csv
```

## Training Models

### Baseline and Lipschitz-Constrained Models

```bash
python scripts/main_train_models.py -db FacebookPagePage -nt gcn -wd 0.0005 -ct full
```

### Non-Negative Constraint

```bash
python scripts/main_train_models.py -db FacebookPagePage -nt gcn -wd 0.0005 -ct positive
```

### Spectral Normalization

```bash
python scripts/main_train_models.py -db FacebookPagePage -nt gcn -wd 0.0005 -ct spectral
```

### Randomized Smoothing

```bash
python scripts/main_train_models_RS.py -db FacebookPagePage -nt gcn -wd 0.0005
```

### Adversarial Training

```bash
python scripts/main_train_models_AT.py -db FacebookPagePage -nt gcn -wd 0.0005
```

## Robustness Evaluation

### Baseline and Lipschitz-Constrained Models

```bash
python scripts/main_attack_models.py -db FacebookPagePage -nt gcn -a apgd_l2_dl -f results_attacks_apgd_l2_dl_lipschitz.csv
python scripts/main_attack_models.py -db FacebookPagePage -nt gcn -a apgd_linf -f results_attacks_apgd_linf_lipschitz.csv -el 0.01 0.05 0.1 0.2 0.3 0.5 0.7 1
```

### Constraint Variants

```bash
python scripts/main_attack_models.py -db FacebookPagePage -nt gcn -ct positive -a apgd_l2_dl -f results_attacks_apgd_l2_dl_lipschitz_positive.csv
python scripts/main_attack_models.py -db FacebookPagePage -nt gcn -ct spectral -a apgd_l2_dl -f results_attacks_apgd_l2_dl_SN.csv
```

### Randomized Smoothing

```bash
python scripts/main_attack_models_RS.py -db FacebookPagePage -nt gcn -a apgd_l2_dl -f results_attacks_apgd_l2_dl_RS.csv
python scripts/main_attack_models_RS.py -db FacebookPagePage -nt gcn -a apgd_linf -f results_attacks_apgd_linf_RS.csv -el 0.01 0.05 0.1 0.2 0.3 0.5 0.7 1
```

### Adversarial Training

```bash
python scripts/main_attack_models_AT.py -db FacebookPagePage -nt gcn -a apgd_l2_dl -f results_attacks_apgd_l2_dl_AT.csv
python scripts/main_attack_models_AT.py -db FacebookPagePage -nt gcn -a apgd_linf -f results_attacks_apgd_linf_AT.csv -el 0.01 0.05 0.1 0.2 0.3 0.5 0.7 1
```

## Graph-Defense Baselines

```bash
python scripts/main_train_attack_graph_defenses.py -wd 0.0005 -a apgd_l2_dl -db FacebookPagePage -nt svd -f results_attacks_apgd_l2_dl_svd.csv
python scripts/main_train_attack_graph_defenses.py -wd 0.0005 -a apgd_l2_dl -db FacebookPagePage -nt jaccard -f results_attacks_apgd_l2_dl_jaccard.csv
python scripts/main_train_attack_graph_defenses.py -wd 0.0005 -a apgd_l2_dl -db FacebookPagePage -nt rgcn -f results_attacks_apgd_l2_dl_rgcn.csv
```

## Attack Comparison

```bash
python scripts/main_attack_models.py -db FacebookPagePage -nt gcn -a apgd_l2_dlr -f results_attacks_apgd_l2_dlr_lipschitz.csv
python scripts/main_attack_models.py -db FacebookPagePage -nt gcn -a apgd_l2_ce -f results_attacks_apgd_l2_ce_lipschitz.csv
python scripts/main_attack_models.py -db FacebookPagePage -nt gcn -a pgd_l2_dlr -f results_attacks_pgd_l2_dlr_lipschitz.csv
python scripts/main_attack_models.py -db FacebookPagePage -nt gcn -a pgd_l2_ce -f results_attacks_pgd_l2_ce_lipschitz.csv
python scripts/main_attack_models.py -db FacebookPagePage -nt gcn -a pgd_l2_dl -f results_attacks_pgd_l2_dl_lipschitz.csv
```

## Plotting Results

### Accuracy and Robustness Trade-Off

```bash
python scripts/main_plot_results.py -cl 28 11 3 -nl gcn -f results_attacks_apgd_l2_dl_lipschitz.csv
```

### Baseline, Lipschitz-Constrained, Randomized Smoothing, and Adversarial Training

```bash
python scripts/main_plot_results.py -cl 11 -sl 0.6 -ael 150 -nl gcn -f results_attacks_apgd_l2_dl_lipschitz.csv -fa results_attacks_apgd_l2_dl_AT.csv -frs results_attacks_apgd_l2_dl_RS.csv
python scripts/main_plot_results.py -cl 11 -sl 0.6 -ael 150 -nl gcn -f results_attacks_apgd_linf_lipschitz.csv -fa results_attacks_apgd_linf_AT.csv -frs results_attacks_apgd_linf_RS.csv
```

### Baseline, Lipschitz-Constrained, and Spectral Normalization

```bash
python scripts/main_plot_results.py -cl 11 -snl 9 -f results_attacks_apgd_l2_dl_lipschitz.csv -fsn results_attacks_apgd_l2_dl_SN.csv
```

### Graph Defenses

```bash
python scripts/main_plot_results.py -cl 11 -jl 0 -svdl 30 -rl 0.1 -f results_attacks_apgd_l2_dl_lipschitz.csv -fj results_attacks_apgd_l2_dl_jaccard.csv -fr results_attacks_apgd_l2_dl_rgcn.csv -fsvd results_attacks_apgd_l2_dl_svd.csv
```

### Different Attacks

```bash
python scripts/main_plot_results.py -cl 11 -nl gcn -fl results_attacks_apgd_l2_dl_lipschitz.csv results_attacks_apgd_l2_dlr_lipschitz.csv results_attacks_apgd_l2_ce_lipschitz.csv results_attacks_pgd_l2_dl_lipschitz.csv results_attacks_pgd_l2_dlr_lipschitz.csv results_attacks_pgd_l2_ce_lipschitz.csv
```

## Generated Outputs

Training, evaluation, and plotting scripts create output files and directories automatically when needed.

| Output | Description |
|---|---|
| `saved_models/` | Trained model checkpoints |
| `saved_history/` | Training histories and intermediate metrics |
| `saved_figures/` | Generated plots |
| `results_attacks_*.csv` | Robustness evaluation results used by the plotting scripts |

## Notes

- Run commands from the repository root, not from inside the `scripts/` directory.
- Run the training command before the corresponding attack command.
- Run the attack command before the corresponding plotting command.
- Make sure the output file names passed to the plotting script match the files generated by the attack scripts.
- If a plot command fails because a CSV file is missing, first run the corresponding robustness evaluation command.
