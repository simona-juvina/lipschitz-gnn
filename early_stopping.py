import numpy as np
import torch
from torch import nn
from typing import Dict, Any, Optional
# user-defined modules
from utils import get_Lips_constant


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, config_params: Dict[str, Any], delta: float = 0, constraint: Optional[float] = None, strategy: str = 'min'):
        """
        Initialize the EarlyStopping callback.

        Args:
            config_params (dict): Configuration parameters.
            delta (float, optional): Minimum change in the monitored quantity to qualify as an improvement. Defaults to 0.
            constraint (float, optional): Lipschitz constraint value. Defaults to None.
            strategy (str, optional): Early stopping strategy. Must be either 'min' or 'convergence'. Defaults to 'min'.
        """
        self.patience = config_params['patience']
        self.path = config_params['espath']

        self.counter = 0
        self.best_score = None
        self.last_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.constraint = constraint
        self.strategy = strategy
        
    def __call__(self, val_loss: float, model: nn.Module):
        """
        Perform early stopping check.

        Args:
            val_loss (float): Validation loss.
            model (nn.Module): Model to be saved.
        """
        if self.strategy == 'convergence':
            score = -val_loss
            if self.constraint:
                current_lip_const = get_Lips_constant(model)
                if current_lip_const - 0.01 < self.constraint:
                    if self.last_score is None:
                        self.last_score = score
                        self.save_checkpoint(val_loss, model)
                    elif abs(score - self.last_score) <= self.delta:
                        self.counter += 1
                        if self.counter >= self.patience:
                            self.early_stop = True
                    else:
                        self.last_score = score
                        self.save_checkpoint(val_loss, model)
                        self.counter = 0
            else:
                if self.last_score is None:
                    self.last_score = score
                    self.save_checkpoint(val_loss, model)
                elif abs(score - self.last_score) <= self.delta:
                    self.counter += 1
                    if self.counter >= self.patience:
                        self.early_stop = True
                else:
                    self.last_score = score
                    self.save_checkpoint(val_loss, model)
                    self.counter = 0          
            
        elif self.strategy == 'min':
            score = -val_loss
            if self.constraint:
                current_lip_const = get_Lips_constant(model)
                if current_lip_const - 0.05 < self.constraint:
                    if self.best_score is None:
                        self.best_score = score
                        self.save_checkpoint(val_loss, model)
                    elif score <= self.best_score + self.delta:
                        self.counter += 1
                        if self.counter >= self.patience:
                            self.early_stop = True
                    else:
                        self.best_score = score
                        self.save_checkpoint(val_loss, model)
                        self.counter = 0
            else:
                if self.best_score is None:
                    self.best_score = score
                    self.save_checkpoint(val_loss, model)
                elif score <= self.best_score + self.delta:
                    self.counter += 1
                    if self.counter >= self.patience:
                        self.early_stop = True
                else:
                    self.best_score = score
                    self.save_checkpoint(val_loss, model)
                    self.counter = 0

    def save_checkpoint(self, val_loss: float, model: nn.Module):
        """
        Saves the model when the validation loss decreases
        Args:
            val_loss (float): The current validation loss.
            model (nn.Module): Model to be saved.
        """
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

        
