import argparse
import os
import csv
import numpy as np
import pandas as pd
import torch
import warnings
import matplotlib.pyplot as plt
from pathlib import Path
from torch import nn
from functools import partial

# user-defined modules
from attacks.pgd import pgd
from attacks.apgd import apgd
from model import GraphNN
from utils import update_parameters, get_Lips_constant_upper, load_dataset
from attacks.utils import run_attack, compute_attack_metrics
from train_model_utils import test_one_epoch


lip_ct       = [30, 28, 26, 24, 20, 18, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0.5]
datasets     = ('FacebookPagePage', 'GitHub', 'LastFMAsia', 'DeezerEurope')
networks     = ('gcn', 'sage')
constr_types = ('full', 'positive', 'spectral')
pert_list    = [25, 50, 100, 200, 400, 600, 800, 1000]

attacks    = {
    'pgd_linf': partial(pgd, norm=float('inf'), num_steps=100, restarts=10, loss_function='dl'),
    'pgd_l2_ce': partial(pgd, norm=2, num_steps=100, restarts=10, loss_function='ce'),
    'pgd_l2_dl': partial(pgd, norm=2, num_steps=100, restarts=10, loss_function='dl'),
    'pgd_l2_dlr': partial(pgd, norm=2, num_steps=100, restarts=10, loss_function='dlr'),
    'apgd_linf': partial(apgd, norm=float('inf'), n_iter=100, n_restarts=10, loss_function='dl'),
    'apgd_l2_dl': partial(apgd, norm=2, n_iter=100, n_restarts=10, loss_function='dl'),
    'apgd_l2_dlr': partial(apgd, norm=2, n_iter=100, n_restarts=10, loss_function='dlr'),
    'apgd_l2_ce': partial(apgd, norm=2, n_iter=100, n_restarts=10, loss_function='ce'),
}

config = {
    'with_constraint': True,
    'norm_each': False,  # Constraint the norm on each node instead of the whole graph
    'print_info': False,
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluate robustness of constrained models')
    
    # Add arguments
    parser.add_argument('--results_filename', '-f', type=str, default='results_attacks_lip.csv', help='Name of csv file to record results')
    parser.add_argument('--db_name', '-db', type=str, help='Dataset to train', default='FacebookPagePage', choices=datasets)
    parser.add_argument('--hidden_dim', '-hd', type=int, nargs='+', help='List of hidden dimensions for the network', default=[16, 16]) 
    parser.add_argument('--network_type', '-nt', type=str, default='gcn', help='Type of network architecture', choices=networks)
    parser.add_argument('--attack', '-a', type=str.lower, default='apgd_l2_dl', help='Attack(s) to use', choices=attacks)
    parser.add_argument('--constraint_type', '-ct', type=str, default='full', help='Constraint type: Lipschiz, positive, or spectral normalization', choices=constr_types)
    parser.add_argument('--constr_list', '-cl', type=float, nargs='+', help='Lipschitz constraint(s)', default=lip_ct) 
    parser.add_argument('--num_splits', '-ns', type=int, default=10, help='Number of seeded splits')
    parser.add_argument('--epsilon_list', '-el', type=float, nargs='+', help='Constraint(s) on the size of the perturbation', default=pert_list)
    parser.add_argument('--num_epochs', '-n', type=int, default=1000, help='Number of epochs')
    parser.add_argument('--gpu_number', '-gn', type=int, default=0, help='GPU core number')
    parser.add_argument('--test_size', '-ts', type=float, default=0.2, help='Test size(percentage)')
    parser.add_argument('--val_size', '-vs', type=float, default=0.2, help='Validation size(percentage)')
    parser.add_argument('--saved_models_path',  '-smp', type=str, default='./saved_models', help='Name of folder to save models')
    args = parser.parse_args()

    args.constr_list = [float(item) for item in args.constr_list]
    # Configuration
    for arg in vars(args):
        config[arg] = getattr(args, arg)
    if (('pos' in args.constraint_type) and ('spectral' not in args.constraint_type)):
        rho_list = ['no', 'pos']
    else:
        rho_list = ['no'] + args.constr_list

    # Device
    device = torch.device('cuda:' + str(config['gpu_number']) if torch.cuda.is_available() else 'cpu')
    config['device'] = device
    torch.cuda.set_device(config['gpu_number']) 

    assert os.path.isdir(args.saved_models_path), f"The folder {args.saved_models_path} does not exist."

    # Write results to file
    perf_metrics = ['attack_success', 'original_acc', 'orig_acc_val', 'after_attack_acc', 'orig_f1_list','after_att_f1_list']
    column_names = ['split_num', 'db_name', 'upper_Lip_ct', 'attack', 'epsilon', 'norm_each_node', 'network_type', 'hidden_dim']
    for pm in perf_metrics:
        column_names += [str(pm) +'_'+str(i) for i in rho_list]
    if (os.path.isfile(args.results_filename) != True):
        with open(args.results_filename,"w",newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(column_names)
    else:
        df_old = pd.read_csv(args.results_filename, delimiter=",")
        assert list(df_old.columns) == column_names,  f"Results file {args.results_filename} exists but the column names are inconsistent i.e. the list of Lipschitz constraints."
        print("\nResults file exists. Appending..\n")

    model_path = f"{config['saved_models_path']}/{args.db_name}{args.network_type.upper()}"

    for epsilon in args.epsilon_list:
        # run attacks and keep track of the best perturbations found
        attack_method = attacks[args.attack]
        attack_method.keywords['eps'] = epsilon
        attack_method.keywords['norm_each_node'] = config['norm_each']
        print(f"{args.attack} attack on {args.network_type.upper()} network with hidden layers {args.hidden_dim} on {args.db_name} dataset for {args.num_splits} splits\nPerturbation = {epsilon}\n")

        for split in range(args.num_splits):
            print(f"Split {split}:")
            # load dataset
            dataset, num_classes, num_features = load_dataset(args.db_name, config, seed=split)
            neurons_per_layer = [num_features] + args.hidden_dim + [num_classes]
            config['neurons_per_layer'] = neurons_per_layer

            list_len = len(rho_list)
            att_success_list   = np.zeros(list_len)
            orig_acc_list      = np.zeros(list_len)
            orig_acc_val_list  = np.zeros(list_len)
            after_att_acc_list = np.zeros(list_len) 
            orig_f1_list       = np.zeros((list_len, num_classes))
            after_att_f1_list  = np.zeros((list_len, num_classes))

            # No constraint
            with_constraint = False
            config = update_parameters(config, with_constraint=with_constraint)
            
            if (('pos' in args.constraint_type) and ('spectral' not in args.constraint_type)):
                ct_list = ['no_', 'pos_']
            else:
                ct_list = ['no_'] + args.constr_list

            for it, rho in enumerate(ct_list):
                # Load model
                if ('spectral' in args.constraint_type and rho != 'no_'):
                    mp = f"{model_path}_{rho}spectral_norm_{str(args.hidden_dim)}_split{str(split)}"
                else:
                    mp = f"{model_path}_{rho}constraint_{str(args.hidden_dim)}_split{str(split)}"
                if (rho == 'no_'):
                    assert Path(mp).exists(), f"Model to attack {mp} does not exist"
                elif not Path(mp).exists():
                    print(f"Constrained lip={rho} model to attack {mp} does not exist")
                    continue

                model = GraphNN(args.network_type, neurons_per_layer).to(device)
                checkpoint = torch.load(mp, map_location=device)
                model.load_state_dict(checkpoint)
                if (rho=='no_'):
                    upper_lip_const = np.round(get_Lips_constant_upper(model), 2)
                    lip_values = [upper_lip_const] + args.constr_list

                # Run attack
                attack_data = run_attack(model=model, dataset=dataset, attack=attack_method)

                attack_metrics = compute_attack_metrics(model=model, attack_data=attack_data)
                clean_acc, adv_acc = attack_metrics['accuracy_orig'], attack_metrics['accuracy_attack']
                print(f'{args.attack.upper()} {rho} constraint - Clean acc: {clean_acc:.2%} - Adv acc: {adv_acc:.2%}')

                δ = attack_data['adv_dataset'].x - attack_data['dataset'].x
                l2 = δ.flatten().norm(p=2, dim=0)
                linf2 = δ.flatten().abs().max()
                print_rho = upper_lip_const if (rho=='no_') else rho
                print(f'{args.attack.upper()} {rho} constraint - L2: {l2:.3g} - Linf,2: {linf2:.3g}')
                print(f"Perturbation effect ratio: {attack_metrics['pert_effect_ratio']:.2f}")

                att_success_list[it]    = attack_metrics['success'].mean()
                orig_acc_list[it]       = attack_metrics['accuracy_orig']
                orig_acc_val_list[it]   = test_one_epoch(dataset, model, num_classes, device, stage='val')['f1_sc_micro'].item()
                after_att_acc_list[it]  = attack_metrics['accuracy_attack']
                orig_f1_list[it,:]      = attack_metrics['f1_score_orig']
                after_att_f1_list[it,:] = attack_metrics['f1_score_attack']

            to_write_perf = list(att_success_list) + list(orig_acc_list) + list(orig_acc_val_list) + list(after_att_acc_list) + list(orig_f1_list) + list(after_att_f1_list)
            to_write_perf = [f"{np.array2string(i, precision=2, floatmode='fixed')}" for i in to_write_perf]

            write_row = [split, args.db_name, upper_lip_const, args.attack, epsilon, config['norm_each'], args.network_type, args.hidden_dim]
            write_row += to_write_perf
            with open(args.results_filename, "a", newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(write_row)
