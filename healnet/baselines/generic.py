"""
Some relatively generic baseline models for comparison - mainly using the regularised FCNN
"""

import torch.nn as nn
import torch
import torch.nn.functional as F
from typing import *

class FCNN(nn.Module):

    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int, dropout: float = 0.5):
        super().__init__()

        # Construct topology
        self.input_layer = nn.Linear(input_size, hidden_sizes[0])
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_sizes[i], hidden_sizes[i+1]) for i in range(len(hidden_sizes)-1)
        ])

        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)

    def forward(self, x):
        # only select second to last dimension
        x = x.squeeze()
        # x = einops.rearrange(x, 'b feat c -> feat c')
        x = self.relu(self.input_layer(x))
        for layer in self.hidden_layers:
            # apply droput and acvitations to hidden layers
            x = self.relu(self.dropout(layer(x)))
        x = self.output_layer(x)
        return x

class RegularizedFCNN(nn.Module):
    def __init__(self, output_dim, dropout_rate=0.2, l1_penalty=0.01, l2_penalty=0.01):
        super(RegularizedFCNN, self).__init__()

        # Store the attributes
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.l1_penalty = l1_penalty
        self.l2_penalty = l2_penalty

        # Placeholder for layers
        self.input_layer = None
        self.hidden_layer = nn.Linear(128, 64)
        self.dropout_layer = nn.Dropout(dropout_rate)
        self.output_layer = nn.Linear(64, output_dim)

    def forward(self, inputs: List[torch.Tensor]):

        if type(inputs) == list:
            inputs = inputs[0]

        # Get the input dimension and create the input layer if it doesn't exist
        if self.input_layer is None:
            input_dim = inputs.shape[1]
            self.input_layer = nn.Linear(input_dim, 128).to(inputs.device)

        x = F.relu(self.input_layer(inputs))
        x = F.relu(self.hidden_layer(x))
        x = self.dropout_layer(x)
        return torch.sigmoid(self.output_layer(x))

    def l1_regularization(self):
        l1_reg = torch.tensor(0., requires_grad=True)
        for name, param in self.named_parameters():
            if 'weight' in name:
                l1_reg = l1_reg + torch.norm(param, 1)
        return self.l1_penalty * l1_reg

    def l2_regularization(self):
        l2_reg = torch.tensor(0., requires_grad=True)
        for name, param in self.named_parameters():
            if 'weight' in name:
                l2_reg = l2_reg + torch.norm(param, 2)
        return self.l2_penalty * l2_reg


