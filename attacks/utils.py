import numpy as np
import torch
import torch_geometric
import torchmetrics
from collections import OrderedDict
from distutils.version import LooseVersion
from functools import partial
from inspect import isclass
from typing import Callable, Dict, Optional, Tuple, Union
from torch import Tensor, nn
from torch.nn import functional as F
from deeprobust.graph.defense import GCNJaccard, GCNSVD, GCN, RGCN
from adv_lib.distances.lp_norms import l0_distances, l1_distances, l2_distances, linf_distances
from adv_lib.utils import BackwardCounter, ForwardCounter


def predict_inputs(model: nn.Module, inputs: Tensor, edge_index: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    if (isinstance(model, (GCNJaccard, GCNSVD))):
        logits = model.forward(inputs, edge_index)[mask]
    elif (isinstance(model, RGCN)):
        model.features = inputs
        logits = model.forward()[mask]
    else:
        logits = model(inputs, edge_index, None)[mask]
            
    probabilities = torch.softmax(logits, 1)
    predictions = logits.argmax(1)
    return logits, probabilities, predictions


def run_attack(model: nn.Module,
               dataset: torch_geometric.data.Data,
               attack: Callable,
               targets: Optional[Tensor] = None) -> dict:
    # get device and target
    device = next(model.parameters()).device
    targeted, adv_labels = False, dataset.y
    if targets is not None:
        targeted, adv_labels = True, targets

    attack_data = {'dataset': dataset, 'targets': adv_labels if targeted else None}

    # move data to device (clone to prevent in-place modifications)
    dataset = dataset.clone().to(device)

    # initialize trackers
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    forward_counter, backward_counter = ForwardCounter(), BackwardCounter()
    model.register_forward_pre_hook(forward_counter)
    if LooseVersion(torch.__version__) >= LooseVersion('1.8'):
        model.register_full_backward_hook(backward_counter)
    else:
        model.register_backward_hook(backward_counter)

    start.record()
    adv_dataset = attack(model, dataset, targeted=targeted, targets=adv_labels)
    # performance monitoring
    end.record()
    torch.cuda.synchronize()
    total_time = (start.elapsed_time(end)) / 1000  # times for cuda Events are in milliseconds
    num_forwards = forward_counter.num_samples_called
    num_backwards = backward_counter.num_samples_called

    if isinstance(attack, partial) and (callback := attack.keywords.get('callback')) is not None:
        callback.reset_windows()

    attack_data.update({
        'adv_dataset': adv_dataset.to(device),
        'time': total_time,
        'num_forwards': num_forwards / len(dataset.x),
        'num_backwards': num_backwards / len(dataset.x),
    })
    return attack_data


_default_metrics = OrderedDict([
    ('linf', linf_distances),
    ('l0', l0_distances),
    ('l1', l1_distances),
    ('l2', l2_distances),
])


def compute_attack_metrics(model: nn.Module,
                           attack_data: Dict[str, Union[Tensor, float]],
                           metrics: Dict[str, Callable] = _default_metrics) -> Dict[str, Union[np.ndarray, float]]:
    dataset, adv_dataset = attack_data['dataset'], attack_data['adv_dataset']
    device = next(model.parameters()).device
    to_device = lambda tensor: tensor.to(device)

    all_predictions = [[] for _ in range(6)]
    distances = {k: [] for k in metrics.keys()}
    metrics = {k: v().to(device) if (isclass(v.func) if isinstance(v, partial) else False) else v for k, v in
               metrics.items()}

    # get predictions on clean and perturbed data
    dataset, adv_dataset = map(to_device, [dataset, adv_dataset])
    edge_index, test_mask = dataset.edge_index, dataset.test_mask
    
    if (isinstance(model, (GCNJaccard, GCNSVD))):
        adj = model.adj_norm
    else:
        adj = edge_index
    clean_preds = predict_inputs(model=model, inputs=dataset.x, edge_index=adj, mask=test_mask)
    adv_preds = predict_inputs(model=model, inputs=adv_dataset.x, edge_index=adj, mask=test_mask)

    clean_preds_x = predict_inputs(model=model, inputs=dataset.x, edge_index=adj, mask=None)
    adv_preds_x = predict_inputs(model=model, inputs=adv_dataset.x, edge_index=adj, mask=None)

    noise_norm = (adv_dataset.x - dataset.x).norm()
    pert_effect_ratio = ((adv_preds_x[0] - clean_preds_x[0]).norm()) / noise_norm

    for metric, metric_func in metrics.items():
        distances[metric].append(metric_func(adv_dataset.x.unsqueeze(0), dataset.x.unsqueeze(0)).detach().cpu())

    logits, probs, preds = clean_preds
    logits_adv, probs_adv, preds_adv = adv_preds
    for metric in metrics.keys():
        distances[metric] = torch.cat(distances[metric], 0)

    test_labels = dataset.y[test_mask]
    accuracy_orig = (preds == test_labels).float().mean().item()
    f1_score_orig = torchmetrics.functional.f1_score(preds, test_labels, average=None, num_classes=max(test_labels) + 1)
    accuracy_attack = (preds_adv == test_labels).float().mean().item()
    f1_score_attack = torchmetrics.functional.f1_score(preds_adv, test_labels, average=None,
                                                       num_classes=max(test_labels) + 1)

    if attack_data['targets'] is not None:
        test_labels = attack_data['targets']
        success = (preds_adv == test_labels)
    else:
        success = (preds_adv != test_labels)

    prob_orig = probs.gather(1, test_labels.unsqueeze(1)).squeeze(1)
    prob_adv = probs_adv.gather(1, test_labels.unsqueeze(1)).squeeze(1)
    labels_infhot = torch.zeros_like(logits_adv).scatter_(1, test_labels.unsqueeze(1), float('inf'))
    real = logits_adv.gather(1, test_labels.unsqueeze(1)).squeeze(1)
    other = (logits_adv - labels_infhot).max(1).values
    diff_vs_max_adv = (real - other)
    nll = F.cross_entropy(logits, test_labels, reduction='none')
    nll_adv = F.cross_entropy(logits_adv, test_labels, reduction='none')

    data = {
        'time': attack_data['time'],
        'num_forwards': attack_data['num_forwards'],
        'num_backwards': attack_data['num_backwards'],
        'targeted': attack_data['targets'] is not None,
        'preds': preds,
        'adv_preds': preds_adv,
        'pert_effect_ratio': pert_effect_ratio.item(),
        'accuracy_orig': accuracy_orig,
        'f1_score_orig': f1_score_orig,
        'success': success,
        'probs_orig': prob_orig,
        'accuracy_attack': accuracy_attack,
        'f1_score_attack': f1_score_attack,
        'probs_adv': prob_adv,
        'logit_diff_adv': diff_vs_max_adv,
        'nll': nll,
        'nll_adv': nll_adv,
        'distances': distances,
    }
    data = {k: v.detach().cpu().numpy() if isinstance(v, Tensor) else v for k, v in data.items()}
    return data


def print_metrics(metrics: dict) -> None:
    np.set_printoptions(formatter={'float': '{:0.3f}'.format}, threshold=16, edgeitems=3,
                        linewidth=120)  # To print arrays with less precision
    success = metrics['success']
    fail = bool(success.mean() != 1)
    print('Perturbation effect ratio: {:.2f}'.format(metrics['pert_effect_ratio']))
    print('Attack success: {:.2%}'.format(success.mean()) + fail * ' - {}'.format(success))
    print('Original accuracy: {:.2%}'.format(metrics['accuracy_orig']))
    print('After attack accuracy: {:.2%}'.format(metrics['accuracy_attack']))
    print(f"Original F1 score: {np.array2string(metrics['f1_score_orig'], precision=2, floatmode='fixed')}")
    print(f"After attack F1 score: {np.array2string(metrics['f1_score_attack'], precision=2, floatmode='fixed')}")
    print('Attack done in: {:.2f}s with {:.4g} forwards and {:.4g} backwards.'.format(
        metrics['time'], metrics['num_forwards'], metrics['num_backwards']))

    for distance, values in metrics['distances'].items():
        data = values.numpy()
        print('{}: {} - Average: {:.3f} - Median: {:.3f}'.format(distance, data, data.mean(), np.median(data)))
    attack_type = 'targets' if metrics['targeted'] else 'correct'
    print('Logit({} class) - max_Logit(other classes): {} - Average: {:.2f}'.format(
        attack_type, metrics['logit_diff_adv'], metrics['logit_diff_adv'].mean()))
    print('NLL of target/pred class: {:.3f}'.format(metrics['nll_adv'].mean()))
