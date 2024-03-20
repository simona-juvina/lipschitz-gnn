import argparse
import os
import torch
import numpy as np
from pathlib import Path
from torch import nn
from functools import partial

# user-defined modules
from attacks.pgd import pgd
from attacks.apgd import apgd
from model import GraphNN
from utils import load_dataset
from attacks.utils import run_attack
from train_model_utils import train_test_model


optimizers    = ('adam', 'sgd')
datasets      = ('FacebookPagePage', 'GitHub', 'LastFMAsia', 'DeezerEurope')
networks      = ('gcn', 'sage')
es_strategies = ('min', 'convergence')
adv_epsilon   = [10, 50, 100, 150, 200, 300, 400, 500, 600, 800, 900, 1000]
activation_functions = ('relu', 'sigmoid', 'tanh', 'leakyrelu', 'silu')

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
    'lr_scheduler': True,
    'measure_to_plot': 'acc',
    'with_constraint': False,
    'constraint_type': 'full' # dummy
}
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser('Adversarial training')
    
    # Add arguments
    parser.add_argument('--db_name', '-db', type=str, help='Dataset to train', default='FacebookPagePage', choices=datasets)
    parser.add_argument('--hidden_dim', '-hd', type=int, nargs='+', help='List of hidden dimensions for the network', default=[16, 16])
    parser.add_argument('--network_type', '-nt', type=str, default='gcn', help='Type of network architecture', choices=networks)
    parser.add_argument('--attack', '-a', type=str.lower, default='apgd_l2_dl', help='Attack(s) to use', choices=attacks)
    parser.add_argument('--norm_each', '-ne', action='store_true', help='Constrain the norm on each node instead of the whole graph')
    parser.add_argument('--adv_epsilon_list', '-ael', type=float, nargs='+', default=adv_epsilon, help='Constraint(s) on the size of the perturbation for AT')
    parser.add_argument('--activation_function', '-af', type=str, help='Activation function', default='relu', choices=activation_functions)
    parser.add_argument('--num_splits', '-ns', type=int, default=10, help='Number of seeded splits')
    parser.add_argument('--gpu_number', '-gn', type=int, default=0, help='GPU core number')
    parser.add_argument('--test_size', '-ts', type=float, default=0.2, help='Test size (percentage)')
    parser.add_argument('--val_size', '-vs', type=float, default=0.2, help='Validation size (percentage)')
    parser.add_argument('--num_epochs', '-n', type=int, default=2000, help='Number of epochs')
    parser.add_argument('--early_stop', '-es', action='store_false', help='Use early stopping')
    parser.add_argument('--patience', '-p', type=int, default=200, help='Number of epochs with no improvement after which training will be stopped')
    parser.add_argument('--optimizer_type', '-o', type=str, default='adam', help='Optimizer', choices=optimizers)
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.01, help='Optimizer learning rate')
    parser.add_argument('--weight_decay', '-wd', type=float, default=0.0005, help='Weight decay for L2 regularization')
    parser.add_argument('--es_strategy', '-ess', type=str, default='min', help='Early stopping strategy', choices=es_strategies)
    parser.add_argument('--es_delta', '-esd', type=float, default=0.000007, help='Minimum improvement for early stopping')
    parser.add_argument('--saved_figures_path', '-sfp', type=str, default='./saved_figures', help='Name of folder to save figures')
    parser.add_argument('--saved_history_path', '-shp', type=str, default='./saved_history', help='Name of folder to save history')
    parser.add_argument('--saved_models_path',  '-smp', type=str, default='./saved_models', help='Name of folder to save models')
    args = parser.parse_args()

    # Configuration
    for arg in vars(args):
        config[arg] = getattr(args, arg)
        
    # Device
    device = torch.device('cuda:' + str(config['gpu_number']) if torch.cuda.is_available() else 'cpu')
    config['device'] = device
    torch.cuda.set_device(config['gpu_number']) 

    # Create directories if they don't exist
    if not os.path.isdir(args.saved_figures_path):
        os.makedirs(args.saved_figures_path)
    if not os.path.isdir(args.saved_history_path):
        os.makedirs(args.saved_history_path)
    if not os.path.isdir(args.saved_models_path):
        os.makedirs(args.saved_models_path)
    
    # Model path
    model_path = f"{args.db_name}{args.network_type.upper()}"

    print(f"Adversarial train models for {args.num_splits} splits. {args.attack} attack on {args.network_type.upper()} network with hidden layers {args.hidden_dim} on {args.db_name} dataset\n") 

    for split in range(args.num_splits):
        print(f"Split {split}:")
        # Load dataset
        dataset, num_classes, num_features = load_dataset(args.db_name, config, seed=split)
        neurons_per_layer = [num_features] + args.hidden_dim + [num_classes]
        config['neurons_per_layer'] = neurons_per_layer

        # Run attacks and keep track of the best perturbations found
        for epsilon in args.adv_epsilon_list:
            i = 0
            attack_method = attacks[args.attack]
            attack_method.keywords['eps'] = epsilon
            attack_method.keywords['norm_each_node'] = args.norm_each
            print(f"Perturbation = {epsilon:g}\n")

            if i == 0:
                # Load model
                mp = f"{config['saved_models_path']}/{model_path}_no_constraint_{str(args.hidden_dim)}_split{str(split)}"
                assert Path(mp).exists(), f"Base model {mp} does not exist"
                model = GraphNN(args.network_type, neurons_per_layer).to(device)
                checkpoint = torch.load(mp, map_location=device)
                model.load_state_dict(checkpoint)

                config['model_path'] = f"{args.db_name}{args.network_type.upper()}_adveps_{epsilon}_{args.attack}"
                i += 1
            # Run attack on train + validation datasets
            attack_method.keywords['pert_mask'] = dataset.train_mask | dataset.val_mask
            attack_data = run_attack(model=model, dataset=dataset, attack=attack_method)

            attack_data['adv_dataset'].train_mask = dataset.train_mask
            attack_data['adv_dataset'].test_mask  = dataset.test_mask
            attack_data['adv_dataset'].val_mask   = dataset.val_mask       

            perf_metrics_no_constr, acc_metrics_test_no_constr, lip_no_constr, model = train_test_model(attack_data['adv_dataset'], config, split, model=model)
