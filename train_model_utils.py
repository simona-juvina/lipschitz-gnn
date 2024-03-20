import time
import numpy as np
import torch
import torch_geometric
from torch import nn
from pathlib import Path
from typing import Optional, Tuple, Any, List, Dict, Union

# user-defined modules
from utils import calculate_metrics_torch, get_Lips_constant, get_Lips_constant_upper, dump_object, load_object
from plot_utils import plot_history
from early_stopping import EarlyStopping
from constraint import Constraint_SAGE, Constraint_GCN
from model import GraphNN

    
def train_one_epoch(data: torch_geometric.data.data.Data,
                    model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer,
                    constraint: Union[Constraint_SAGE,Constraint_GCN],
                    num_classes: int, 
                    device: torch.device,
                    epoch: int, 
                    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None) -> Dict[str, Any]:
    """Train the model on a single epoch.

    Args:
        data: The graph data.
        model: The model to be trained.
        optimizer: The optimizer for updating model parameters.
        constraint: The constraint object.
        num_classes: The number of classes.
        device: The device for computation.
        epoch: The current epoch.
        scheduler: The learning rate scheduler (optional).

    Returns:
        A dictionary containing the accuracy metrics.
    """
    model.train()  # Prepare the model for training

    optimizer.zero_grad()  # Clear the gradients of all optimized variables
    data.to(device)

    train_mask = data.train_mask

    # Compute the predicted values and target labels for train nodes
    predicted = model(data.x, data.edge_index, data.edge_weight)[train_mask]
    target = data.y[train_mask]

    # Save previous weights
    num_layers = len(model.gcn_stack)
    old_weights = [None] * num_layers
    if model.network_type == 'sage':
        for i, gcn_block in enumerate(model.gcn_stack):
            w0 = torch.clone(gcn_block.lin_r.weight.detach().data)
            w1 = torch.clone(gcn_block.lin_l.weight.detach().data)
            old_weights[i] = [w0, w1]
    elif model.network_type == 'gcn':
        for i, gcn_block in enumerate(model.gcn_stack):
            w = torch.clone(gcn_block.lin.weight.detach().data)
            old_weights[i] = w

    # Calculate the loss
    loss = torch.nn.CrossEntropyLoss()(predicted, target)
    predicted = predicted.argmax(1)

    # Backward pass: compute gradient of the loss with respect to model parameters
    loss.backward()

    # Perform a single optimization step (parameter update)
    optimizer.step()

    # Adjust learning rate
    if scheduler is not None:
        scheduler.step(loss)

    # Apply constraint
    constraint.on_batch_end(old_weights)

    # Calculate accuracy metrics
    acc_metrics = calculate_metrics_torch(target, predicted, num_classes)
    acc_metrics['loss'] = loss.item()

    return acc_metrics


def test_one_epoch(data: torch_geometric.data.data.Data, 
                   model: torch.nn.Module, 
                   num_classes: int,
                   device: torch.device, 
                   stage: str) -> Dict[str, float]:
    """Test the model on a single epoch.

    Args:
        data: The graph data.
        model: The trained model.
        num_classes: The number of classes.
        device: The device for computation.
        stage: The stage for testing: validation/test.

    Returns:
        A dictionary containing the accuracy metrics.
    """
    model.eval()  # Prepare the model for evaluation
    
    data.to(device)
    mask = getattr(data, f"{stage}_mask")

    # Compute the predicted values and target labels
    predicted = model(data.x, data.edge_index, data.edge_weight)[mask]
    target = data.y[mask]

    # Record the loss
    loss = torch.nn.CrossEntropyLoss()(predicted, target).item()
    predicted = predicted.argmax(1)

    # Calculate accuracy metrics
    acc_metrics = calculate_metrics_torch(target, predicted, num_classes)
    acc_metrics['loss'] = loss

    return acc_metrics


def train_model(data: torch_geometric.data.data.Data,
                model: torch.nn.Module, 
                config_params: Dict[str, Any], 
                constraint: Union[Constraint_SAGE,Constraint_GCN], 
                num_classes: int) -> Tuple:
    """Train the model.

    Args:
        data: The graph data.
        model: The model to be trained.
        config_params: Configuration parameters for training.
        constraint: The constraint object.
        num_classes: The number of classes.

    Returns:
        A tuple containing the performance metrics and the trained model.
    """
    num_epochs = config_params['num_epochs']
    early_stopping = config_params['early_stop']
    espath = config_params['espath']
    patience = config_params['patience']
    optimizer = config_params['optimizer']
    scheduler = config_params['lr_scheduler']
    device = config_params['device']

    if not constraint.with_constraint or "pos" in constraint.constraint_type.lower():
        constraint.rho = None

    metrics_train = []
    metrics_val = []
    if early_stopping:
        # Initialize the early_stopping object
        es = EarlyStopping(config_params=config_params, constraint=constraint.rho, strategy=config_params['es_strategy'], delta=config_params['es_delta'])

    torch.cuda.synchronize()
    for t in range(num_epochs):
        torch.cuda.synchronize()
        start_time_epoch = time.time()
        acc_metrics_train = train_one_epoch(data, model, optimizer, constraint, num_classes, device, t, scheduler)
        acc_metrics_valid = test_one_epoch(data, model, num_classes, device, stage='val')

        metrics_train.append(acc_metrics_train)
        metrics_val.append(acc_metrics_valid)

        torch.cuda.synchronize()
        end_time_epoch = time.time()
        elapsed = end_time_epoch - start_time_epoch

        if config_params['print_results'] and t % config_params['print_every_k'] == 0:
            print(f"Epoch {t+1}/{num_epochs} - {elapsed:>0.4f} s")
            print(f"Train:      Avg loss: {acc_metrics_train['loss']:>4f}, Acc: {acc_metrics_train['acc']:>4f}, Accuracy: {acc_metrics_train['acc']:>4f}")
            print(f"Validation: Avg loss: {acc_metrics_valid['loss']:>4f}, Acc: {acc_metrics_valid['acc']:>4f}, Accuracy: {acc_metrics_valid['acc']:>4f}")

        if early_stopping:
            # Early stopping needs the validation loss to check if it has decreased,
            # and if it has, it will make a checkpoint of the current model
            es(acc_metrics_valid["loss"], model)

            if es.early_stop:
                print(f"Early stopping at epoch {t}")
                break

    if early_stopping:
        # Load the last checkpoint with the best model
        model.load_state_dict(torch.load(espath))
    else:
        current_lip_const = get_Lips_constant(model)
        torch.save(model.state_dict(), espath)

    perf_dict = {
        'train': metrics_train,
        'val': metrics_val
    }
    return perf_dict, model


def train_test_model(dataset: torch_geometric.data.data.Data, 
                     config_params: Dict[str, Any],
                     split: int,
                     model: Optional[torch.nn.Module] = None) -> Tuple:
    """
    Train and test the model.

    Args:
        dataset: The graph data.
        config_params: Configuration parameters for training and testing.
        split: The split number.
        model: The pre-trained model (optional).

    Returns:
        A tuple containing the performance metrics, test metrics, Lipschitz constant, and the trained model.
    """
    # Get configuration parameters
    db_name = config_params['db_name']
    device = config_params['device']
    network_type = config_params['network_type']
    patience = config_params['patience']

    if not config_params['early_stop']:
        patience = 1
        
    if (config_params['with_constraint'] == False):
        print_constr = 'no_constraint'
        rho = None
    else:
        if ('full' in config_params['constraint_type'].lower()):
            print_constr = 'constraint'
            rho     = config_params['rho']
            old_rho = config_params['old_rho']
        elif ('spectral' in config_params['constraint_type'].lower()):
            if ('pos' in config_params['constraint_type'].lower()):
                print_constr = 'spectral_norm_pos'
            else:
                print_constr = 'spectral_norm'
            rho     = config_params['rho']
            old_rho = config_params['old_rho']
        elif ('pos' in config_params['constraint_type'].lower()):
            print_constr = 'pos_constraint'
            rho = ''
            old_rho = ''
    old_perf_metrics = None
    
    hidden_dim = config_params['neurons_per_layer'][1:-1]
    config_params['num_layers'] = len(hidden_dim) + 1

    if (print_constr != 'no_constraint'):
        save_string = f"{config_params['model_path']}_{rho}{print_constr}_{str(hidden_dim)}_split{str(split)}"
        history_path            = f"{config_params['saved_history_path']}/{save_string}"
        config_params['espath'] = f"{config_params['saved_models_path']}/{save_string}"

        save_string_old = f"{config_params['model_path']}_{old_rho}{print_constr}_{str(hidden_dim)}_split{str(split)}"
        old_history_path = f"{config_params['saved_history_path']}/{save_string_old}"
        old_model_path   = f"{config_params['saved_models_path']}/{save_string_old}"
    else:
        save_string = f"{config_params['model_path']}_{print_constr}_{str(hidden_dim)}_split{str(split)}"
        history_path            = f"{config_params['saved_history_path']}/{save_string}"
        config_params['espath'] = f"{config_params['saved_models_path']}/{save_string}"


    num_classes = config_params['neurons_per_layer'][-1]
    
    if not model:
        if config_params['continue_training'] and print_constr == 'constr':
            try:
                model = GraphNN(network_type, config_params['neurons_per_layer'], activation=config_params['activation_function']).to(device)
                model.load_state_dict(torch.load(old_model_path, map_location=device))
                old_perf_metrics = load_object(old_history_path)
            except: 
                print(f'Old model with lip = {old_rho} does not exist. Training model from scratch to Lip={rho}')
        else:
            train_val_dataset = dataset.x[dataset.train_mask + dataset.val_mask].cpu()
            mean_tv = train_val_dataset.mean(axis=0)
            std_tv = train_val_dataset.std(axis=0)

            model = GraphNN(network_type, config_params['neurons_per_layer'], mean=mean_tv, std=std_tv, activation=config_params['activation_function']).to(device)

            if print_constr == 'constr':
                print(f"Training from scratch to Lip={rho}")

    # Constraint
    if network_type == 'sage':
        constr = Constraint_SAGE(model, config_params, device=device,
                                 with_constraint=config_params['with_constraint'], constraint_type=config_params['constraint_type'])
    elif network_type == 'gcn':
        constr = Constraint_GCN(model, config_params, device=device,
                                with_constraint=config_params['with_constraint'], constraint_type=config_params['constraint_type'])

    # Loss and optimization function
    if config_params['optimizer_type'].lower() == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config_params['learning_rate'], weight_decay=config_params['weight_decay'])
    elif config_params['optimizer_type'].lower() == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=config_params['learning_rate'], weight_decay=config_params['weight_decay'], momentum=0.9)

    # Learning rate scheduler
    if config_params['lr_scheduler'] is not None:
        config_params['lr_scheduler'] = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True, factor=0.1, patience=100)

    config_params['optimizer'] = optimizer

    # Train network
    perf_metrics, model = train_model(dataset, model, config_params, constr, num_classes)
    acc_metrics_test = test_one_epoch(dataset, model, num_classes, device, stage='test')

    if print_constr == 'no_constraint':
        theta_bar = get_Lips_constant_upper(model)
    else:
        theta_bar = get_Lips_constant(model)

    # Save performance metrics
    dump_object(history_path, perf_metrics)

    if config_params['print_results']:
        print(f"\n{print_constr.upper()} PERFORMANCE")
        print(f"train accuracy: {perf_metrics['train'][-patience]['acc']:.4f}")
        print(f"valid accuracy: {perf_metrics['val'][-patience]['acc']:.4f}")
        print(f"test accuracy: {acc_metrics_test['acc']:.4f}")
        print(f"test accuracy: {acc_metrics_test['f1_sc']}")
        print("----------------------------------------------------------------------------------------------")

    if config_params['plot']:
        fig_name = f"{config_params['saved_figures_path']}/training_{save_string}"
        plot_history(perf_metrics, lip_constr=theta_bar, fig_name=fig_name, perf_measure=config_params['measure_to_plot'])

    return perf_metrics, acc_metrics_test, theta_bar, model
