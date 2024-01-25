import torch
from torch import Tensor
import torch.nn as nn
import einops
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Callable, Tuple, Optional
from torchvision.models import resnet18, ResNet18_Weights



class MultiModEncoder(nn.Module, ABC):
    """Abstract encoder for MultiModN"""

    def __init__(self, state_size: int):
        super(MultiModEncoder, self).__init__()
        self.state_size = state_size

    @abstractmethod
    def forward(self, state: Tensor, x: Tensor) -> Tensor:
        pass




class MLPEncoder(MultiModEncoder):
    """Multi-layer perceptron encoder"""
    def __init__(
            self,
            state_size: int,
            n_features: int,
            hidden_layers: Tuple[int],
            activation: Callable = F.relu,
            device: Optional[torch.device] = None,
    ):
        super().__init__(state_size)

        self.activation = activation

        dim_layers = [n_features] + list(hidden_layers) + [self.state_size, ]

        self.layers = nn.ModuleList()
        for i, (in_dim, out_dim) in enumerate(zip(dim_layers, dim_layers[1:])):
            # The state is concatenated to the input of the last layer
            if i == len(dim_layers) - 2:
                self.layers.append(
                    nn.Linear(in_dim + self.state_size, out_dim, device=device))
            else:
                self.layers.append(nn.Linear(in_dim, out_dim, device=device))

    def forward(self, state: Tensor, x: Tensor) -> Tensor:
        # b, *_ = x.shape
        # state = einops.repeat(state, "d -> b d", b=b)

        for layer in self.layers[0:-1]:
            x = self.activation(layer(x))

        output = self.layers[-1](torch.cat([x, state], dim=1))

        # reduce state over batch
        # output = nn.Parameter(einops.reduce(output, "b d -> d", "mean"))

        return nn.Parameter(output)


class PatchEncoder(MultiModEncoder):
    """RNN encoder adjusted for patched images"""

    def __init__(
        self,
        state_size: int,
        n_features: int,
        hidden_layers: Tuple[int],
        activation: Callable = F.relu,
    ):
        super().__init__(state_size)

        self.activation = activation

        dim_layers = [n_features] + list(hidden_layers) + [self.state_size,]

        self.layers = nn.ModuleList()
        for i, (inDim, outDim) in enumerate(zip(dim_layers, dim_layers[1:])):
            # The state is concatenated to the input of the last layer
            if i == len(dim_layers)-2:
                self.layers.append(nn.RNN(inDim + self.state_size, outDim, batch_first=True))
            else:
                self.layers.append(nn.RNN(inDim, outDim, batch_first=True))

    def forward(self, state: Tensor, x: Tensor) -> Tensor:
        # expand state
        # b, *_ = x.shape
        # state = einops.repeat(state, "d -> b d", b=b)

        for layer in self.layers[:-1]:
            out, h_n = layer(x)
            x = self.activation(out)

        # need to average over patches
        output, h_n = self.layers[-1](torch.cat([einops.reduce(tensor=x, pattern="b c d -> b d", reduction="sum"), state], dim=1))

        # reduce state over batch
        # output = nn.Parameter(einops.reduce(output, "b d -> d", "mean"))

        return nn.Parameter(output)



class ResNet(nn.Module):
    def __init__(self, *, state_size=0, freeze=False, pretrained_path=None, pretrained=True):
        super().__init__()

        if pretrained_path is not None and pretrained:
            raise ValueError(
                "Loading a pretrained ResNet should either be from torch.vision (pretrained=True) "
                "or from a checkpoint (pretrained_path) but not both."
            )

        # if pretrained, loads ResNet18 pretrained on ImageNet
        # self.resnet = models.resnet18(pretrained=pretrained)
        self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.state_size = state_size

        self.fc = nn.Linear(512 + self.state_size, self.state_size)

        # load pre-trained ResNet from path
        if pretrained_path:
            model_dict = self.resnet.state_dict()
            # filter out unnecessary keys
            pretrained_dict = {
                k: v for k, v in torch.load(pretrained_path).items() if k in model_dict
            }
            # overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            # load the new state dict
            self.resnet.load_state_dict(model_dict)

        # remove final classification layer
        self.resnet.fc = nn.Identity()

        if freeze:
            for p in self.resnet.parameters():
                p.requires_grad = False

    def forward(self, state, x):
        # expand state

        representations = self.resnet(x)
        output = self.fc(torch.cat([representations, state], dim=1))

        return nn.Parameter(output)
