"""
Configuring weights and biases tracking/setup
"""
import wandb
from typing import *
import random

def wb_tracking(config: Dict) -> None:

    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="healnet",

        # track hyperparameters and run metadata
        config=config
        # {
        # "learning_rate": 0.02,
        # "architecture": "CNN",
        # "dataset": "CIFAR-100",
        # "epochs": 10,
        # }
    )