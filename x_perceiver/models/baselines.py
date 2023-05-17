import torch.nn as nn
from typing import List
import einops
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

#
# class ConvNet(nn.Module):
#
#     def __init__(self, input_size: List[int], hidden_sizes, output_size: int):
#         super().__init__()
#
#         self.conv1 = nn.Conv2d(input_size[0], hidden_sizes[0], kernel_size=5)
#         self.conv2 = nn.Conv2d(hidden_sizes[0], hidden_sizes[1], kernel_size=5)
#         self.conv2_drop = nn.Dropout2d()
#         self.fc1 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
#         self.fc2 = nn.Linear(hidden_sizes[2], output_size)

