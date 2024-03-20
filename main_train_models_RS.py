import argparse
import torch
import os

# user-defined modules
from utils import load_dataset
from train_model_utils import train_test_model

datasets      = ('FacebookPagePage', 'GitHub', 'LastFMAsia', 'DeezerEurope')
networks      = ('gcn', 'sage')
es_strategies = ('min', 'convergence')
sigmas        = (0.1, 0.25, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.4, 1.6, 1.8, 2)   
optimizers    = ('adam', 'sgd')
activation_functions = ('relu', 'sigmoid', 'tanh', 'leakyrelu', 'silu')

config = {
    'print_info': False,
    'print_results': True,
    'print_every_k': 200,
    'plot': True,
    'lr_scheduler': True,
    'measure_to_plot': 'acc',
    'continue_training': False, # dummy
    'constraint_type': 'full', # dummy
    'with_constraint': False
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train models for randomized smoothing')

    # Add arguments
    parser.add_argument('--db_name', '-db', type=str, help='Dataset to train', default='FacebookPagePage', choices=datasets)
    parser.add_argument('--hidden_dim', '-hd', type=int, nargs='+', help='List of hidden dimensions for the network', default=[16, 16])
    parser.add_argument('--network_type', '-nt', type=str, default='gcn', help='Type of network architecture', choices=networks)
    parser.add_argument('--sigma_list', '-sl', type=float, nargs='+', default=sigmas, help='Sigma randomized smoothing')
    parser.add_argument('--activation_function', '-af', type=str, help='Activation function', default='relu', choices=activation_functions)
    parser.add_argument('--num_splits', '-ns', type=int, default=10, help='Number of seeded splits')
    parser.add_argument('--test_size', '-ts', type=float, default=0.2, help='Test size (percentage)')
    parser.add_argument('--val_size', '-vs', type=float, default=0.2, help='Validation size (percentage)')
    parser.add_argument('--num_epochs', '-ne', type=int, default=2000, help='Number of epochs')
    parser.add_argument('--early_stop', '-es', action='store_false', help='Use early stopping')
    parser.add_argument('--patience', '-p', type=int, default=200, help='Number of epochs with no improvement after which training will be stopped')
    parser.add_argument('--gpu_number', '-gn', type=int, default=0, help='GPU core number')
    parser.add_argument('--optimizer_type', '-o', type=str, default='adam', help='Optimizer', choices=optimizers)
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.01, help='Optimizer learning rate')
    parser.add_argument('--weight_decay', '-wd', type=float, default=0.0005, help='Weight decay for L2 regularization')
    parser.add_argument('--dropout', '-d', type=float, default=0, help='Dropout')
    parser.add_argument('--es_strategy', '-ess', type=str, default='min', help='Early stopping strategy', choices=es_strategies)
    parser.add_argument('--es_delta', '-esd', type=float, default=0.000007, help='Minimum improvement for early stopping')
    parser.add_argument('--saved_figures_path', '-sfp', type=str, default='./saved_figures', help='Name of folder to save figures')
    parser.add_argument('--saved_history_path', '-shp', type=str, default='./saved_history', help='Name of folder to save history')
    parser.add_argument('--saved_models_path',  '-smp', type=str, default='./saved_models', help='Name of folder to save models')
    args = parser.parse_args()

    args.sigma_list = [float(item) for item in args.sigma_list]
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
        
    for sigma in args.sigma_list:
        # Model path
        config['model_path'] = f"{args.db_name}{args.network_type.upper()}_sigma{sigma}"
        print(f"Training {args.network_type.upper()} network with hidden layers {args.hidden_dim} and sigma = {sigma} on {args.db_name} dataset for {args.num_splits} splits")

        for split in range(args.num_splits):
            # Load dataset
            dataset, num_classes, num_features = load_dataset(args.db_name, config, seed=split)
            dataset = dataset.to(device)
            config['neurons_per_layer'] = [num_features] + args.hidden_dim + [num_classes]

            # Apply Gaussian noise to training and validation data
            dataset.x[dataset.train_mask] += torch.randn_like(dataset.x, device=device)[dataset.train_mask] * sigma
            dataset.x[dataset.val_mask]   += torch.randn_like(dataset.x, device=device)[dataset.val_mask] * sigma

            # Train and test the model
            perf_metrics_no_constr, acc_metrics_test_no_constr, lip_no_constr, model = train_test_model(dataset, config, split)
