import argparse
import torch
import os
from torch import nn

# user-defined modules
from utils import update_parameters, load_dataset
from train_model_utils import train_test_model    
    
optimizers    = ('adam', 'sgd')
datasets      = ('FacebookPagePage', 'GitHub', 'DeezerEurope', 'LastFMAsia')
networks      = ('gcn', 'sage')
es_strategies = ('min', 'convergence')
constr_types  = ('full', 'positive', 'spectral')
lip_ct        = [30, 28, 26, 24, 20, 18, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0.5]
activation_functions = ('relu', 'sigmoid', 'tanh', 'leakyrelu', 'silu')

config = {
    'print_info': False,
    'print_results': True,
    'print_every_k': 200,
    'plot': True,
    'lr_scheduler': True,
    'measure_to_plot': 'f1',
    'continue_training': False,
    'with_constraint': True,
    'nit': 100,
    'criterion': 100,
    'cnst': 0.01,
    'alpha': 2.1
}

if __name__ == '__main__':
    """
    Trains models based on command-line arguments.
    """
    # Argument Parser
    parser = argparse.ArgumentParser('Train models')
    parser.add_argument('--db_name', '-db', type=str, help='Dataset to train', default='FacebookPagePage', choices=datasets)
    parser.add_argument('--hidden_dim', '-hd', type=int, nargs='+', help='List of hidden dimensions for the network', default=[16, 16]) 
    parser.add_argument('--network_type', '-nt', type=str, default='gcn', help='Type of network architecture', choices=networks)
    parser.add_argument('--constraint_type', '-ct', type=str, default='full', help='Constraint type: Lipschiz, positive, or spectral normalization', choices=constr_types)
    parser.add_argument('--constr_list', '-cl', type=float, nargs='+', help='Lipschitz constraint(s)', default=lip_ct) 
    parser.add_argument('--num_splits', '-ns', type=int, default=10, help='Number of seeded splits')
    parser.add_argument('--test_size', '-ts', type=float, default=0.2, help='Test size (percentage)')
    parser.add_argument('--val_size', '-vs', type=float, default=0.2, help='Validation size (percentage)')
    parser.add_argument('--num_epochs', '-n', type=int, default=2000, help='Number of epochs')
    parser.add_argument('--early_stop', '-es', action='store_false', help='Use early stopping')
    parser.add_argument('--patience', '-p', type=int, default=200, help='Number of epochs with no improvement after which training will be stopped')
    parser.add_argument('--activation_function', '-af', type=str, help='Activation function', default='relu', choices=activation_functions)
    parser.add_argument('--optimizer_type', '-o', type=str, default='adam', help='Optimizer', choices=optimizers)
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.01, help='Optimizer learning rate')
    parser.add_argument('--weight_decay', '-wd', type=float, default=0.0005, help='Weight decay for L2 regularization')
    parser.add_argument('--gpu_number', '-gn', type=int, default=0, help='GPU core number')
    parser.add_argument('--es_strategy', '-ess', type=str, default='min', help='Early stopping strategy', choices=es_strategies)
    parser.add_argument('--es_delta', '-esd', type=float, default=0.000007, help='Minimum improvement for early stopping')
    parser.add_argument('--saved_figures_path', '-sfp', type=str, default='./saved_figures', help='Name of folder to save figures')
    parser.add_argument('--saved_history_path', '-shp', type=str, default='./saved_history', help='Name of folder to save history')
    parser.add_argument('--saved_models_path',  '-smp', type=str, default='./saved_models', help='Name of folder to save models')
    args = parser.parse_args()

    args.constr_list = [float(item) for item in args.constr_list]
    # Configuration
    for arg in vars(args):
        config[arg] = getattr(args, arg)
    constraint_type = config['constraint_type']
    config['rho'] = 0  # dummy value
    config['old_rho'] = 0  # dummy value
    
    if (('pos' in constraint_type) and ('spectral' not in constraint_type)):
        rho_list = ['no', 'pos']
    else:
        rho_list = ['no'] + args.constr_list
  
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
    config['model_path'] = f"{args.db_name}{args.network_type.upper()}"
    print(f"Training {args.network_type.upper()} network with hidden layers {args.hidden_dim} on {args.db_name} dataset for {args.num_splits} splits")

    for split in range(config['num_splits']):
        # Load dataset
        dataset, num_classes, num_features = load_dataset(args.db_name, config, seed=split)
        config['neurons_per_layer'] = [num_features] + args.hidden_dim + [num_classes]

        # No constraint
        config = update_parameters(config, with_constraint=False, rho=0)
        perf_metrics_no_constr, acc_metrics_test_no_constr, lip_no_constr, model = train_test_model(dataset, config, split)

        # Lipschitz constraint
        if (('pos' in constraint_type) and ('spectral' not in constraint_type)):
            constr_list2 = ['no_', 'pos_']
        else:
            constr_list2 = ['no_'] + config['constr_list']
                    
        old_rho = constr_list2[0]
        rho = constr_list2[0]
        for i in range(len(constr_list2) - 1):
            old_rho = rho
            # Don't train a model with a lower Lipschitz constant than lip_no_constr
            if (constr_list2[i+1] != 'pos_') and (constr_list2[i+1] > lip_no_constr):
                print(f"Do not train model with lip={constr_list2[i+1]}. Upper constant for the unconstrained model={lip_no_constr:.2f}")
                continue
            rho = constr_list2[i+1]
            print(f'Training with {constraint_type} constraint to constraint = {rho}')
            config = update_parameters(config, with_constraint=True, constraint_type=constraint_type, rho=rho, old_rho=old_rho, continue_training=config['continue_training'])
            perf_metrics_constr, acc_metrics_test_constr, lip_constr, model = train_test_model(dataset, config, split)