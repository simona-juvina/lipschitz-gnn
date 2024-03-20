import argparse
import os
import csv
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch import nn
from functools import partial

# user-defined modules
from attacks.pgd import pgd
from attacks.apgd import apgd
from attacks.utils import run_attack
from model import GraphNN
from utils import load_dataset, Smooth, calculate_metrics_torch


sigmas    = (0.1, 0.25, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.4, 1.6, 1.8, 2)   
datasets  = ('FacebookPagePage', 'GitHub', 'LastFMAsia', 'DeezerEurope')
networks  = ('gcn', 'sage')
pert_list = [25, 50, 100, 200, 400, 600, 800, 1000]

attacks    = {
    'pgd_linf': partial(pgd, norm=float('inf'), n_iter=100, restarts=10, loss_function='dl'),
    'pgd_l2_ce': partial(pgd, norm=2, num_steps=100, restarts=10, loss_function='ce'),
    'pgd_l2_dl': partial(pgd, norm=2, num_steps=100, restarts=10, loss_function='dl'),
    'pgd_l2_dlr': partial(pgd, norm=2, num_steps=100, restarts=10, loss_function='dlr'),
    'apgd_linf': partial(apgd, norm=float('inf'), n_iter=100, n_restarts=100, loss_function='dl'),
    'apgd_l2_dl': partial(apgd, norm=2, n_iter=100, n_restarts=10, loss_function='dl'),
    'apgd_l2_dlr': partial(apgd, norm=2, n_iter=100, n_restarts=10, loss_function='dlr'),
    'apgd_l2_ce': partial(apgd, norm=2, n_iter=100, n_restarts=10, loss_function='ce'),
}

config = {
    'print_info': False,
    'print_results': True,
    'print_every_k': 200,
    'plot': True,
    'measure_to_plot': 'acc',
    'with_constraint': False
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluate robustness of randomized smoothing models')

    # Add the arguments
    parser.add_argument('--results_filename', '-f', type=str, default='results_attacks_RS.csv', help='Name of csv file to record results')
    parser.add_argument('--db_name', '-db', type=str, help='Dataset to train', default='FacebookPagePage', choices=datasets)
    parser.add_argument('--hidden_dim', '-hd', type=int, nargs='+', help='List of hidden dimensions for the network', default=[16, 16])
    parser.add_argument('--network_type', '-nt', type=str, default='gcn', help='Type of network architecture', choices=networks)
    parser.add_argument('--attack', '-a', type=str.lower, default='apgd_l2_dl', help='Attack(s) to use', choices=attacks)
    parser.add_argument('--norm_each', '-ne', action='store_true', help='Constrain the norm on each node instead of the whole graph')
    parser.add_argument('--sigma_list', '-sl', type=float, nargs='+', default=sigmas, help='Sigma randomized smoothing')
    parser.add_argument('--num_splits', '-ns', type=int, default=10, help='Number of seeded splits')
    parser.add_argument('--epsilon_list', '-el', type=float, nargs='+', help='Constraint(s) on the size of the perturbation', default=pert_list)
    parser.add_argument('--gpu_number', '-gn', type=int, default=0, help='GPU core number')
    parser.add_argument('--test_size', '-ts', type=float, default=0.2, help='Test size (percentage)')
    parser.add_argument('--val_size', '-vs', type=float, default=0.2, help='Validation size (percentage)')
    parser.add_argument('--saved_models_path', '-smp', type=str, default='./saved_models', help='Name of folder to save models')
    parser.add_argument("--N", type=int, default=1000, help="Number of samples to use")
    parser.add_argument("--alpha", '-al', type=float, default=0.001, help="Failure probability")
    args = parser.parse_args()

    args.sigma_list = [float(item) for item in args.sigma_list]

    # Configuration
    for arg in vars(args):
        config[arg] = getattr(args, arg)

    # Device
    device = torch.device('cuda:' + str(config['gpu_number']) if torch.cuda.is_available() else 'cpu')
    config['device'] = device
    torch.cuda.set_device(config['gpu_number']) 

    assert os.path.isdir(args.saved_models_path),  f"The folder {args.saved_models_path} does not exist."

    # Define performance metrics and column names
    perf_metrics = ['original_acc', 'after_attack_acc', 'orig_f1_list', 'after_att_f1_list']
    column_names = ['split_num', 'db_name', 'attack', 'epsilon', 'norm_each_node', 'network_type', 'hidden_dim']
    for pm in perf_metrics:
        column_names += [str(pm) + '_' + str(i) for i in args.sigma_list]

    if not os.path.isfile(args.results_filename):
        # Create a new file and write column names
        with open(args.results_filename, "w", newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(column_names)
    else:
        # Read existing file and check column names consistency
        df_old = pd.read_csv(args.results_filename, delimiter=",")
        assert list(df_old.columns) == column_names, f"Results file {args.results_filename} exists but the column names are inconsistent i.e. the list of σ values."
        print("\nResults file exists. Appending..\n")
    
    for epsilon in args.epsilon_list:
        # Run attacks and keep track of the best perturbations found
        attack_method = attacks[args.attack]

        attack_method.keywords['eps'] = epsilon
        attack_method.keywords['norm_each_node'] = args.norm_each

        print(f"{args.attack} attack on {args.network_type.upper()} network with hidden layers {args.hidden_dim} on {args.db_name} dataset for {args.num_splits} splits\nPerturbation = {epsilon}")

        for split in range(args.num_splits):
            print(f"Split {split}:")
            # Load dataset
            dataset, num_classes, num_features = load_dataset(args.db_name, config, seed=split)
            neurons_per_layer = [num_features] + args.hidden_dim + [num_classes]
            config['neurons_per_layer'] = neurons_per_layer

            list_len = len(args.sigma_list)
            orig_acc_list = np.zeros(list_len)
            after_att_acc_list = np.zeros(list_len)
            orig_f1_list = np.zeros((list_len, num_classes))
            after_att_f1_list = np.zeros((list_len, num_classes))

            for it, sigma in enumerate(args.sigma_list):
                # Load model
                mp = f"{args.saved_models_path}/{args.db_name}{args.network_type.upper()}_sigma{sigma}_no_constraint_{str(args.hidden_dim)}_split{str(split)}"
                if not Path(mp).exists():
                    print(f"Model {mp} does not exist")
                    continue

                base_classifier = GraphNN(args.network_type, config['neurons_per_layer']).to(device)
                checkpoint = torch.load(mp, map_location=device)
                base_classifier.load_state_dict(checkpoint)

                # create the smooothed classifier g
                smoothed_classifier = Smooth(base_classifier, num_classes, sigma, device)

                # Run attack
                attack_data = run_attack(model=base_classifier, dataset=dataset, attack=attack_method)

                δ = attack_data['adv_dataset'].x.to(device) - attack_data['dataset'].x.to(device)

                adv_dataset = attack_data['adv_dataset'].to(device)
                dataset = dataset.to(device)
                adv_dataset.test_mask = dataset.test_mask

                prediction_clean = torch.tensor(smoothed_classifier.predict(dataset, args.N, args.alpha)).to(device)
                prediction_adv = torch.tensor(smoothed_classifier.predict(adv_dataset, args.N, args.alpha)).to(device)

                metrics_clean = calculate_metrics_torch(dataset.y[dataset.test_mask], prediction_clean, num_classes)
                metrics_adv = calculate_metrics_torch(dataset.y[dataset.test_mask], prediction_adv, num_classes)

                print(f"{args.attack.upper()} {sigma} sigma - Clean acc: {metrics_clean['acc']:.2%} - Adv acc: {metrics_adv['acc']:.2%}")
                orig_acc_list[it] = metrics_clean['acc'].cpu().numpy()
                after_att_acc_list[it] = metrics_adv['acc'].cpu().numpy()
                orig_f1_list[it, :] = metrics_clean['f1_sc'].cpu().numpy()
                after_att_f1_list[it, :] = metrics_adv['f1_sc'].cpu().numpy()
                
            to_write_perf = list(orig_acc_list) + list(after_att_acc_list) + list(orig_f1_list) + list(after_att_f1_list)
            to_write_perf = [f"{np.array2string(i, precision=2, floatmode='fixed')}" for i in to_write_perf]

            write_row = [split, args.db_name, args.attack, epsilon, args.norm_each, args.network_type, args.hidden_dim]
            write_row += to_write_perf
            with open(args.results_filename, "a", newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(write_row)
