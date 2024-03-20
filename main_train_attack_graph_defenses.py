import argparse
import os, csv
import numpy as np
import pandas as pd
import torch
import warnings
import scipy.sparse as sp
import matplotlib.pyplot as plt
from pathlib import Path
from torch import nn
from functools import partial
from deeprobust.graph.defense import GCNJaccard, GCNSVD, GCN, RGCN

# user-defined modules
from utils import load_dataset
from attacks.pgd import pgd
from attacks.apgd import apgd
from attacks.utils import compute_attack_metrics, run_attack


def mask_to_index(mask):
    return torch.nonzero(mask == 1).flatten()
    
pert_list   = [25, 50, 100, 200, 400, 600, 800, 1000]
datasets    = ('FacebookPagePage', 'GitHub', 'DeezerEurope', 'LastFMAsia')
constr_svd  = [30, 25, 20, 15, 10, 5]
constr_jacc = [0, 0.05, 0.1, 0.3, 0.5, 0.7]
constr_rgcn = [0.1, 0.3, 0.5, 0.7, 1, 2, 5]
networks    = ('svd', 'jaccard', 'rgcn')

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
    'print_info': False,
    'print_results': True,
    'print_every_k': 200,
    'plot': True,
    'measure_to_plot': 'acc',
    'with_constraint': False
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Attack models')

    parser.add_argument('--constr_param', '-cl', type=float, nargs='+', help='Constraint(s)') 
    parser.add_argument('--results_filename', '-f', type=str, default='results_attacks_graph_defenses.csv', help='Name of csv file to record results')
    parser.add_argument('--db_name', '-db', type=str, help='Dataset to train', default='FacebookPagePage', choices=datasets)
    parser.add_argument('--hidden_dim', '-hd', type=int, default=16, help='Number of hidden units')
    parser.add_argument('--network_type', '-nt', type=str, default='svd', help='Type of network(s)', choices=networks)
    parser.add_argument('--attack', '-a', type=str.lower, default='apgd_l2_dl', help='Attack(s) to use', choices=attacks)
    parser.add_argument('--num_splits', '-ns', type=int, default=10, help='Number of seeded splits')
    parser.add_argument('--epsilon_list', '-el', type=int, nargs='+', help='Constraint(s) on the size of the perturbation', default=pert_list)
    parser.add_argument('--test_size', '-ts', type=float, default=0.2, help='Test size(percentage)')
    parser.add_argument('--val_size', '-vs', type=float, default=0.2, help='Validation size(percentage)')
    parser.add_argument('--early_stop', '-es', action='store_false', help='Use early stopping')
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.01, help='Optimizer learning rate')
    parser.add_argument('--weight_decay', '-wd', type=float, default=0, help='Weight decay for L2 regularization')
    parser.add_argument('--num_epochs', '-n', type=int, default=1000, help='Number of epochs')
    parser.add_argument('--norm_each', '-ne', action='store_true', help='Constrain the norm on each node instead of the whole graph')
    parser.add_argument('--gpu_number', '-gn', type=int, default=0, help='GPU core number')
    parser.add_argument('--saved_models_path',  '-smp', type=str, default='./saved_models', help='Name of folder to save models')
    
    args = parser.parse_args()

    network_type = args.network_type
    for arg in vars(args):
        if ((arg == 'constr_param') & (getattr(args, arg) == None)):
            if (network_type == 'svd'):
                args.constr_param = constr_svd
            elif (network_type == 'jaccard'):
                args.constr_param = constr_jacc 
            elif (network_type == 'rgcn'):
                args.constr_param = constr_rgcn  
        elif (arg not in ['dataset_list', 'network_list']):
            config[arg] = getattr(args, arg)
       
    # Device
    device = torch.device('cuda:'+ str(config['gpu_number']) if torch.cuda.is_available() else 'cpu')
    config['device'] = device
    torch.cuda.set_device(config['gpu_number']) 

    # Write results to file
    perf_metrics = ['attack_success', 'original_acc', 'orig_acc_val', 'after_attack_acc']
    column_names = ['split_num', 'db_name', 'attack', 'epsilon', 'norm_each_node', 'network_type', 'hidden_dim']
    for pm in perf_metrics:
        column_names += [str(pm) +'_'+str(i) for i in args.constr_param]
    if (os.path.isfile(args.results_filename) != True):
        with open(args.results_filename,"w",newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(column_names)
    else:
        df_old = pd.read_csv(args.results_filename, delimiter=",")
        assert list(df_old.columns) == column_names,  f"Results file {args.results_filename} exists but the column names are inconsistent"
        print("\nResults file exists. Appending..\n")
    
    for split in range(args.num_splits):
        print(f"Split {split}:")
        # load dataset
        dataset, num_classes, num_features = load_dataset(config['db_name'], config, seed=split)
        neurons_per_layer = [num_features] + [args.hidden_dim] + [num_classes]
        config['neurons_per_layer'] = neurons_per_layer
        n = dataset.x.shape[0]
        adj = sp.csr_matrix((np.ones(dataset.edge_index.shape[1]), (dataset.edge_index[0].cpu(),
                                                                    dataset.edge_index[1].cpu())), shape=(n, n))
        features, labels = dataset.x.numpy(), dataset.y.numpy()
        idx_train, idx_val, idx_test = mask_to_index(dataset.train_mask), mask_to_index(dataset.val_mask), mask_to_index(dataset.test_mask)

        list_len = len(args.constr_param)
        eps_len = len(args.epsilon_list)
        att_success_list   = np.zeros((eps_len, list_len))
        orig_acc_list      = np.zeros((eps_len, list_len))
        orig_acc_val_list  = np.zeros((eps_len, list_len))
        after_att_acc_list = np.zeros((eps_len, list_len)) 

        for it, k in enumerate(args.constr_param):
            if (network_type == 'svd'):
                model = GCNSVD(nfeat=features.shape[1], nclass=labels.max()+1, nhid=args.hidden_dim,
                           device=device, weight_decay=args.weight_decay, lr=args.learning_rate).to(device)
                model.fit(features, adj, labels, idx_train, idx_val, k=k, verbose=False)

            elif (network_type == 'jaccard'):
                model = GCNJaccard(nfeat=features.shape[1], nclass=labels.max()+1, nhid=args.hidden_dim, 
                                   device=device, weight_decay=args.weight_decay, 
                                   lr=args.learning_rate,binary_feature=False).to(device)
                model.fit(features, adj, labels, idx_train, idx_val, verbose=False, threshold=k)

            elif (network_type == 'rgcn'):
                features = sp.csc_matrix(features)
                model = RGCN(nnodes=adj.shape[0], nfeat=features.shape[1], nclass=labels.max()+1,
                             nhid=args.hidden_dim, device=device, lr=args.learning_rate, gamma=k).to(device)
                model.fit(features, adj, labels, idx_train, idx_val, verbose=False)


            dicteps = {}
            for epsilon in args.epsilon_list:
                dicteps[epsilon] = []
                # run attacks and keep track of the best perturbations found
                attack_method = attacks[args.attack]

                attack_method.keywords['eps'] = epsilon
                attack_method.keywords['norm_each_node'] = args.norm_each
                print(f"\n{args.attack} attack on {network_type.upper()} network on {config['db_name']} dataset for {args.num_splits} splits\nPerturbation = {epsilon}")

                # Run attack
                attack_data = run_attack(model=model, dataset=dataset, attack=attack_method)
                attack_metrics = compute_attack_metrics(model=model, attack_data=attack_data)

                clean_acc, adv_acc = attack_metrics['accuracy_orig'], attack_metrics['accuracy_attack']
                print(f'{args.attack.upper()} Clean acc: {clean_acc:.2%} - Adv acc: {adv_acc:.2%}')

                δ = attack_data['adv_dataset'].x - attack_data['dataset'].x
                l2 = δ.flatten().norm(p=2, dim=0)
                linf2 = δ.flatten().abs().max()
                print(f'{args.attack.upper()} {k} constraint - L2: {l2:.3g} - Linf,2: {linf2:.3g}')
                print(f"Perturbation effect ratio: {attack_metrics['pert_effect_ratio']:.2f}")

                et = args.epsilon_list.index(epsilon)
                att_success_list[et][it]    = attack_metrics['success'].mean()
                orig_acc_list[et][it]       = attack_metrics['accuracy_orig']
                orig_acc_val_list[et][it]   = model.test(idx_test)
                after_att_acc_list[et][it]  = attack_metrics['accuracy_attack']


        for i in range(eps_len):
            to_write_perf = list(att_success_list[i]) + list(orig_acc_list[i]) + list(orig_acc_val_list[i]) + list(after_att_acc_list[i])
            to_write_perf = [f"{np.array2string(i, precision=2, floatmode='fixed')}" for i in to_write_perf]

            write_row = [split, config['db_name'], args.attack, args.epsilon_list[i], args.norm_each, network_type, args.hidden_dim]
            write_row += to_write_perf
            with open(args.results_filename,"a",newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(write_row)   

            to_write_perf = list(att_success_list) + list(orig_acc_list) + list(orig_acc_val_list) + list(after_att_acc_list) 
            to_write_perf = [f"{np.array2string(i, precision=2, floatmode='fixed')}" for i in to_write_perf]
