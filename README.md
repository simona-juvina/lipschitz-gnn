## Training Graph Neural Networks Subject to a Tight Lipschitz Constraint

We propose a strategy for training a wide range of graph neural networks (GNNs) under tight Lipschitz bound constraints. We proposed a constrained-optimization approach to control the constant, ensuring robustness to adversarial perturbations. We focus on defending against attacks that perturb features while keeping the topology of the graph constant. 

### Examples of running the scripts for the Facebook dataset, for the GCN architecture:
#### Training the models:
Training the baseline and Lipschitz-constrained models:
```
python main_train_models.py -db FacebookPagePage -nt gcn -wd 0.0005 -ct full 
```
Training models with the non-negativity constraint:
```
python main_train_models.py -db FacebookPagePage -nt gcn -wd 0.0005 -ct positive 
```
Training the SN models:
```
python main_train_models.py -db FacebookPagePage -nt gcn -wd 0.0005 -ct spectral 
```
Training the RS models:
```
python main_train_models_RS.py -db FacebookPagePage -nt gcn -wd 0.0005
```
Training the AT models:
```
python main_train_models_AT.py -db FacebookPagePage -nt gcn -wd 0.0005
```

#### Evaluate the robustness of the models:

Evaluate the robustness of the baseline and Lipschitz-constrained models:
- $L_2$ APGD-DL attack
```
python main_attack_models.py -db FacebookPagePage -nt gcn -a apgd_l2_dl -f results_attacks_apgd_l2_dl_lipschitz.csv
```
- $L_{inf}$ APGD-DL attack
```
python main_attack_models.py -db FacebookPagePage -nt gcn -a apgd_linf -f results_attacks_apgd_linf_lipschitz.csv -el 0.01 0.05 0.1 0.2 0.3 0.5 0.7 1 
```
Evaluate the robustness of the models with the non-negativity constraint:
```
python main_attack_models.py -db FacebookPagePage -nt gcn -ct positive -a apgd_l2_dl -f results_attacks_apgd_l2_dl_lipschitz_positive.csv 
```
Evaluate the robustness of the SN models:
```
python main_attack_models.py -db FacebookPagePage -nt gcn -ct spectral -a apgd_l2_dl -f results_attacks_apgd_l2_dl_SN.csv 
```
Evaluate the robustness of the RS models:
- $L_2$ APGD-DL attack
```
python main_attack_models_RS.py -db FacebookPagePage -nt gcn -a apgd_l2_dl -f results_attacks_apgd_l2_dl_RS.csv 
```
- $L_{inf}$ APGD-DL attack
```
python main_attack_models_RS.py -db FacebookPagePage -nt gcn -a apgd_linf -f results_attacks_apgd_linf_RS.csv -el 0.01 0.05 0.1 0.2 0.3 0.5 0.7 1 
```
Evaluate the robustness of the AT models:
- $L_2$ APGD-DL attack
```
python main_attack_models_AT.py -db FacebookPagePage -nt gcn -a apgd_l2_dl -f results_attacks_apgd_l2_dl_AT.csv 
```
- $L_{inf}$ APGD-DL attack
```
python main_attack_models_AT.py -db FacebookPagePage -nt gcn -a apgd_linf -f results_attacks_apgd_linf_AT.csv -el 0.01 0.05 0.1 0.2 0.3 0.5 0.7 1 
```
Train and evaluate the robustness of SVD-GCN, GCN-Jaccard, RGCN:
```
python main_train_attack_graph_defenses.py -wd 0.0005 -a apgd_l2_dl -db FacebookPagePage -nt svd     -f results_attacks_apgd_l2_dl_svd.csv
python main_train_attack_graph_defenses.py -wd 0.0005 -a apgd_l2_dl -db FacebookPagePage -nt jaccard -f results_attacks_apgd_l2_dl_jaccard.csv
python main_train_attack_graph_defenses.py -wd 0.0005 -a apgd_l2_dl -db FacebookPagePage -nt rgcn    -f results_attacks_apgd_l2_dl_rgcn.csv
```
Evaluate the efficiency of different attacks:
```
python main_attack_models.py -db FacebookPagePage -nt gcn -a apgd_l2_dlr -f results_attacks_apgd_l2_dlr_lipschitz.csv 
python main_attack_models.py -db FacebookPagePage -nt gcn -a apgd_l2_ce  -f results_attacks_apgd_l2_ce_lipschitz.csv 
python main_attack_models.py -db FacebookPagePage -nt gcn -a pgd_l2_dlr  -f results_attacks_pgd_l2_dlr_lipschitz.csv 
python main_attack_models.py -db FacebookPagePage -nt gcn -a pgd_l2_ce   -f results_attacks_pgd_l2_ce_lipschitz.csv 
python main_attack_models.py -db FacebookPagePage -nt gcn -a pgd_l2_dl   -f results_attacks_pgd_l2_dl_lipschitz.csv
```

#### Plot the results:
Accuracy - robustness tradeoff:
```
python main_plot_results.py -cl 28 11 3 -nl gcn -f results_attacks_apgd_l2_dl_lipschitz.csv
```
Robustness comparison - baseline, our method, RS, AT:
- $L_2$ APGD-DL attacks
```
python main_plot_results.py -cl 11 -sl 0.6 -ael 150 -nl gcn -f results_attacks_apgd_l2_dl_lipschitz.csv -fa results_attacks_apgd_l2_dl_AT.csv -frs results_attacks_apgd_l2_dl_RS.csv
```
- $L_{inf}$ APGD-DL attack
```
python main_plot_results.py -cl 11 -sl 0.6 -ael 150 -nl gcn -f results_attacks_apgd_linf_lipschitz.csv -fa results_attacks_apgd_linf_AT.csv -frs results_attacks_apgd_linf_RS.csv
```
Robustness comparison - baseline, our method, SN:
```
python main_plot_results.py -cl 11 -snl 9 -f results_attacks_apgd_l2_dl_lipschitz.csv -fsn results_attacks_apgd_l2_dl_SN.csv
```
Robustness comparison - baseline, our method, SVD-GCN, GCN-Jaccard, RGCN:
```
python main_plot_results.py -cl 11 -jl 0 -svdl 30 -rl 0.1 -f results_attacks_apgd_l2_dl_lipschitz.csv -fj results_attacks_apgd_l2_dl_jaccard.csv -fr results_attacks_apgd_l2_dl_rgcn.csv -fsvd results_attacks_apgd_l2_dl_svd.csv
```
Comparison between different attacks:
```
python main_plot_results.py -cl 11 -nl gcn -fl results_attacks_apgd_l2_dl_lipschitz.csv results_attacks_apgd_l2_dlr_lipschitz.csv results_attacks_apgd_l2_ce_lipschitz.csv results_attacks_pgd_l2_dl_lipschitz.csv results_attacks_pgd_l2_dlr_lipschitz.csv results_attacks_pgd_l2_ce_lipschitz.csv 
```

