# Adapted from https://github.com/fra31/auto-attack
import math
import numbers
from functools import partial
from typing import Tuple, Optional, Union
import torch
import torch_geometric
from torch import nn, Tensor
from torch.nn import functional as F
from adv_lib.utils.losses import difference_of_logits_ratio,difference_of_logits
from deeprobust.graph.defense import GCNJaccard, GCNSVD, GCN, RGCN


def apgd(model: nn.Module,
         dataset: torch_geometric.data.Data,
         eps: Union[float, Tensor],
         norm: float,
         norm_each_node: bool = False,
         targeted: bool = False,
         pert_mask: Tensor = None,
         targets: Optional[Tensor] = None,
         n_iter: int = 100,
         n_restarts: int = 10,
         loss_function: str = 'dlr',
         eot_iter: int = 1,
         rho: float = 0.75,
         best_loss: bool = False) -> Tensor:
    """
    Auto-PGD (APGD) attack from https://arxiv.org/abs/2003.01690 with L1 variant from https://arxiv.org/abs/2103.01208.

    Parameters
    ----------
    model : nn.Module
        Model to attack.
    inputs : Tensor
        Inputs to attack. Should be in [0, 1].
    labels : Tensor
        Labels corresponding to the inputs if untargeted, else target labels.
    eps : float or Tensor
        Maximum norm for the adversarial perturbation. Can be a float used for all samples or a Tensor containing the
        distance for each corresponding sample.
    norm : float
        Norm corresponding to eps in {2, float('inf')}.
    targeted : bool
        Whether to perform a targeted attack or not.
    n_iter : int
        Number of optimization steps.
    n_restarts : int
        Number of random restarts for the attack.
    loss_function : str
        Loss to optimize in ['ce', 'dlr', 'dl'].
    eot_iter : int
        Number of iterations for expectation over transformation.
    rho : float
        Parameters for decreasing the step size.
    Returns
    -------
    adv dataset : torch_geometric.data.Data
        Modified dataset to be adversarial to the model.
    """
    assert norm in [2, float('inf')]

    adv_inputs = dataset.x
    adv_percent = 0
    device = next(model.parameters()).device

    apgd_attack = partial(_apgd, model=model, norm=norm, targeted=targeted, loss_function=loss_function,
                          eot_iter=eot_iter, rho=rho)
    for _ in range(n_restarts):
        adv_percent_run, adv_inputs_run = apgd_attack(dataset=dataset, eps=eps, n_iter=n_iter)
        if adv_percent_run >= adv_percent:
            adv_percent = adv_percent_run
            adv_inputs = adv_inputs_run
    return torch_geometric.data.Data(x=adv_inputs, edge_index=dataset.edge_index, y=dataset.y).to(device)


def check_oscillation(loss_steps: Tensor, j: int, k: int, k3: float = 0.75) -> Tensor:
    t = torch.zeros_like(loss_steps[0])
    for counter5 in range(k):
        t.add_(loss_steps[j - counter5] > loss_steps[j - counter5 - 1])
    return t <= k * k3


def _apgd(model: nn.Module,
          dataset: torch_geometric.data.Data,
          eps: Tensor,
          norm: float,
          x_init: Optional[Tensor] = None,
          pert_mask: Tensor = None,
          targeted: bool = False,
          n_iter: int = 100,
          loss_function: str = 'dlr',
          eot_iter: int = 1,
          rho: float = 0.75) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    _loss_functions = {
        'ce': (nn.CrossEntropyLoss(reduction='none'), -1 if targeted else 1),
        'dl': (difference_of_logits, -1),
        'dlr': (partial(difference_of_logits_ratio, targeted=targeted), 1 if targeted else -1),
    }
    device = next(model.parameters()).device
    inputs = dataset.x.clone()
    labels = dataset.y.clone()
    batch_size = len(inputs)
    
    criterion_indiv, multiplier = _loss_functions[loss_function.lower()]
    n_iter_2, n_iter_min, size_decr = max(int(0.22 * n_iter), 1), max(int(0.06 * n_iter), 1), max(int(0.03 * n_iter), 1)
  
    best_percent = 0
    δ_mask = dataset.test_mask if pert_mask is None else pert_mask
    zero_mask = ~δ_mask
    label_mask = dataset.test_mask 
    
    lower, upper = (inputs - eps).clamp_(min=0, max=1), (inputs + eps).clamp_(min=0, max=1)

    if x_init is not None:
        x_adv = x_init.clone()
    elif norm == float('inf'):
        delta = 2 * torch.rand_like(inputs) - 1
        delta[zero_mask] = 0
        x_adv = inputs + delta * (eps / delta.flatten(1).norm(p=float('inf')))
    elif norm == 2:
        delta = torch.randn_like(inputs)
        delta[zero_mask] = 0
        delta.unsqueeze(0).renorm_(p=2, dim=0, maxnorm=eps)
        x_adv = inputs + delta * (eps / delta.flatten().norm(p=2, dim=0))

    x_best = x_adv.clone()
    x_best_adv = inputs.clone()
    
    loss_steps = torch.zeros(n_iter, batch_size, device=device)
    loss_best_steps = torch.zeros(n_iter + 1, batch_size, device=device)

    x_adv.requires_grad_()
    grad = torch.zeros_like(inputs)
    for _ in range(eot_iter):
        if (isinstance(model, (GCNJaccard, GCNSVD))):
            adj = model.adj_norm
            logits = model.forward(x_adv, adj)
        elif (isinstance(model, RGCN)):
            model.features = x_adv
            logits = model.forward()
        else:
            logits = model(x_adv, dataset.edge_index, None)
        
        loss_indiv = multiplier * criterion_indiv(logits, labels)
        grad.add_(torch.autograd.grad(loss_indiv[label_mask].sum(), x_adv, only_inputs=True)[0])
        grad[zero_mask] = 0

    grad.div_(eot_iter)
    grad_best = grad.clone()
    x_adv.detach_()

    node_is_adv = (logits.argmax(1) == labels) if targeted else (logits.argmax(1) != labels)
    loss_best = loss_indiv.detach().clone()
    
    lower, upper = (inputs - eps), (inputs + eps)
    
    alpha = 2 if norm in [2, float('inf')] else 1 if norm == 1 else 2e-2
    step_size = alpha * eps
    x_adv_old = x_adv.clone()
    k = n_iter_2
    counter3 = 0
    
    loss_best_last_check = loss_best.clone()
    reduced_last_check = torch.zeros_like(loss_best, dtype=torch.bool)

    for i in range(n_iter):
        # gradient step
        grad2 = x_adv - x_adv_old
        x_adv_old = x_adv

        a = 0.75 if i else 1.0
              
        if norm == 2:
            grad_norm = grad.flatten(start_dim=0).norm(p=2)
            delta = x_adv + grad.mul_((step_size /grad_norm.add_(1e-12)))
            delta.sub_(inputs)
            delta_norm = delta.flatten().norm(p=2).add_(1e-12)
            x_adv_1 = delta.mul_((min(delta_norm, eps)/(delta_norm))).add_(inputs)

            # momentum
            delta = x_adv.add(x_adv_1 - x_adv, alpha=a).add_(grad2, alpha=1 - a)
            delta.sub_(inputs)
            delta_norm = delta.flatten().norm(p=2).add_(1e-12)
            x_adv_1 = delta.mul_((min(delta_norm, eps)/(delta_norm))).add_(inputs)
            
        elif norm == float('inf'):
            x_adv_1 = (x_adv + step_size * torch.sign(grad))
            x_adv_1 = torch.min(torch.max(x_adv_1, lower), upper)
            # momentum
            x_adv_1 = (x_adv + (x_adv_1 - x_adv) * a + grad2 * (1 - a))
            x_adv_1 = torch.min(torch.max(x_adv_1, lower), upper)
            
        x_adv = x_adv_1
            
        # get gradient
        x_adv.requires_grad_(True)
        grad.zero_()
        for _ in range(eot_iter):
            if (isinstance(model, (GCNJaccard, GCNSVD))):
                adj = model.adj_norm
                logits = model.forward(x_adv, adj)
            elif (isinstance(model, RGCN)):
                model.features = x_adv
                logits = model.forward()
            else:
                logits = model(x_adv, dataset.edge_index, None)
          
            loss_indiv = multiplier * criterion_indiv(logits, labels)
            grad.add_(torch.autograd.grad(loss_indiv[label_mask].sum(), x_adv, only_inputs=True)[0])
            grad[zero_mask] = 0

        grad.div_(eot_iter)
        x_adv.detach_(), loss_indiv.detach_()
        is_adv = (logits.argmax(1) == labels) if targeted else (logits.argmax(1) != labels)
        
        node_is_adv.logical_or_(is_adv)
        x_best_adv = x_adv
        
        adv_percent = node_is_adv[label_mask].flatten().float().mean()

        # check step size
        loss_steps[i] = loss_indiv
        ind = loss_indiv > loss_best
        x_best[ind] = x_adv[ind]
        grad_best[ind] = grad[ind]
        loss_best[ind] = loss_indiv[ind]
        loss_best_steps[i + 1] = loss_best

        counter3 += 1
        if counter3 == k:
            fl_reduce_no_impr = (~reduced_last_check) & (loss_best_last_check >= loss_best)
            reduced_last_check = check_oscillation(loss_steps, i, k, k3=rho) | fl_reduce_no_impr
            loss_best_last_check = loss_best

            if reduced_last_check.any():
                step_size /= 2.0
            k = max(k - size_decr, n_iter_min)

            counter3 = 0

    return adv_percent, x_best_adv

