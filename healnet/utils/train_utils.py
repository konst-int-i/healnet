from typing import *
import torch
import torch.nn as nn

def calc_reg_loss(model, l1: float, model_topo: str, sources: List[str]):

    if model_topo == "fcnn": # don't regularise FCNN
        reg_loss = 0
    elif model_topo == "mcat" and sources == ["omic"]:
        reg_loss = 0
    else:
        l1_norm = sum(p.abs().sum() for p in model.parameters())
        reg_loss = float(l1) * l1_norm
    return reg_loss


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class EarlyStopping:
    def __init__(self, patience=5, verbose=False, mode='min'):
        """
        Constructor for early stopping.

        Parameters:
        - patience (int): How many epochs to wait before stopping once performance stops improving.
        - verbose (bool): If True, prints out a message for each validation metric improvement.
        - mode (str): One of ['min', 'max']. Minimize (e.g., loss) or maximize (e.g., accuracy) the metric.
        """
        assert mode in ['min', 'max'], "Mode must be 'min' or 'max'"
        self.patience = patience
        self.verbose = verbose
        self.counter = 0

        if mode == 'min':
            self.best_metric = float('inf')
            self.operator = torch.lt
        else:
            self.best_metric = float('-inf')
            self.operator = torch.gt

        self.best_model_weights = None
        self.should_stop = False

    def step(self, metric, model):
        """
        Check the early stopping conditions.

        Parameters:
        - metric (float): The latest validation metric (loss, accuracy, etc.).
        - model (torch.nn.Module): The model being trained.

        Returns:
        - bool: True if early stopping conditions met, False otherwise.
        """
        if type(metric) == float: # convert to tensor if necessary
            metric = torch.tensor(metric)

        if self.operator(metric, self.best_metric):
            if self.verbose:
                print(f"Validation metric improved from {self.best_metric:.4f} to {metric:.4f}. Saving model weights.")
            self.best_metric = metric
            self.counter = 0
            self.best_model_weights = model.state_dict().copy()
        else:
            self.counter += 1
            if self.verbose:
                print(f"Validation metric did not improve. Patience: {self.counter}/{self.patience}.")
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop

    def load_best_weights(self, model):
        """
        Load the best model weights.

        Parameters:
        - model (torch.nn.Module): The model to which the best weights should be loaded.
        """
        if self.verbose:
            print(f"Loading best model weights with validation metric value: {self.best_metric:.4f}")
        model.load_state_dict(self.best_model_weights)
        return model
