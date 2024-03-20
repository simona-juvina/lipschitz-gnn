import torch
import torch_geometric
from functools import partial
from typing import Optional, Tuple
from torch import Tensor, nn
from torch.autograd import grad
from torch.nn import functional as F
from deeprobust.graph.defense import GCNJaccard, GCNSVD, GCN, RGCN
from adv_lib.utils.losses import difference_of_logits, difference_of_logits_ratio

def pgd(model: nn.Module,
        dataset: torch_geometric.data.Data,
        eps: float,
        targets: Optional[Tensor] = None,
        targeted: bool = False,
        masks: Tensor = None,
        pert_mask: Tensor = None,
        norm: float = float('inf'),
        norm_each_node: bool = False,
        num_steps: int = 40,
        random_init: bool = False,
        restarts: int = 1,
        loss_function: str = 'ce',
        relative_step_size: float = 0.1 / 3,
        absolute_step_size: Optional[float] = None) -> torch_geometric.data.Data:
    adv_features = dataset.x
    adv_percent = 0
    pgd_attack = partial(_pgd, model=model, ε=eps, targeted=targeted, masks=masks, norm=norm, num_steps=num_steps,
                         loss_function=loss_function, norm_each_node=norm_each_node,
                         relative_step_size=relative_step_size, absolute_step_size=absolute_step_size)

    for i in range(restarts):
        adv_percent_run, adv_features_run = pgd_attack(dataset=dataset, random_init=random_init or (i != 0))
        if adv_percent_run >= adv_percent:
            adv_percent = adv_percent_run
            adv_features = adv_features_run

    return torch_geometric.data.Data(x=adv_features, edge_index=dataset.edge_index, y=dataset.y)


def _pgd(model: nn.Module,
         dataset: torch_geometric.data.Data,
         ε: float,
         targeted: bool = False,
         masks: Tensor = None,
         pert_mask: Tensor = None,
         norm: float = float('inf'),
         norm_each_node: bool = False,
         num_steps: int = 40,
         random_init: bool = False,
         loss_function: str = 'ce',
         relative_step_size: float = 0.1 / 3,
         absolute_step_size: Optional[float] = None) -> Tuple[Tensor, Tensor]:
    _loss_functions = {
        'ce': (partial(F.cross_entropy, reduction='none'), 1),
        'dl': (difference_of_logits, -1),
        'dlr': (partial(difference_of_logits_ratio, targeted=targeted), -1),
    }
    loss_func, multiplier = _loss_functions[loss_function.lower()]
    if targeted:
        multiplier *= -1

    step_size = ε * relative_step_size if absolute_step_size is None else absolute_step_size

    δ = torch.zeros_like(dataset.x, requires_grad=True)
    best_percent = 0
    best_adv = dataset.x

    δ_mask = dataset.test_mask if pert_mask is None else pert_mask
    zero_mask = ~δ_mask
    label_mask = dataset.test_mask if masks is None else masks

    if random_init:
        if norm == float('inf'):
            δ.data.uniform_(-ε, ε)
            δ.data[zero_mask] = 0
        elif norm == 2:
            δ.data.normal_()
            δ.data[zero_mask] = 0
            if norm_each_node:
                δ.data.renorm_(p=2, dim=0, maxnorm=ε)
            else:
                δ.data.unsqueeze(0).renorm_(p=2, dim=0, maxnorm=ε)

    # Pad logits if not all classes are used for test
    const_pad = nn.ConstantPad1d((0, int((dataset.y.max() - dataset.y[label_mask].max()).item())), float('-inf'))

    for i in range(num_steps):
        adv_inputs = dataset.x + δ
        
        if (isinstance(model, (GCNJaccard, GCNSVD))):
            adj = model.adj_norm
            logits = model.forward(adv_inputs, adj)
        elif (isinstance(model, RGCN)):
            model.features = adv_inputs
            logits = model.forward()
        else:
            logits = model(adv_inputs, dataset.edge_index, None)
        
        logits = const_pad(logits)
        if i == 0 and loss_function.lower() in ['dl', 'dlr']:
            labels_infhot = torch.zeros_like(logits).scatter(1, dataset.y.unsqueeze(1), float('inf'))
            loss_func = partial(loss_func, labels_infhot=labels_infhot)

        loss = multiplier * loss_func(logits, dataset.y)[label_mask]
        δ_grad = grad(loss.sum(), δ, only_inputs=True)[0]

        pred = logits.argmax(dim=1)
        node_is_adv = (pred == dataset.y) if targeted else (pred != dataset.y)
        adv_percent = node_is_adv[label_mask].flatten().float().mean()
        if adv_percent >= best_percent:
            best_percent = adv_percent
            best_adv = adv_inputs.detach()

        δ_grad[zero_mask] = 0
        if norm == float('inf'):
            δ.data.add_(δ_grad.sign_(), alpha=step_size).clamp_(min=-ε, max=ε)
        elif norm == 2:
            dim = 1 if norm_each_node else 0
            grad_norm = δ_grad.flatten(start_dim=dim).norm(p=2, dim=dim, keepdim=norm_each_node)
            δ.data.addcdiv_(δ_grad, grad_norm.clamp_min_(1e-6), value=step_size)
            if norm_each_node:
                δ.data.renorm_(p=2, dim=0, maxnorm=ε)
            else:
                δ.data.unsqueeze(0).renorm_(p=2, dim=0, maxnorm=ε)

    return best_percent, best_adv


