import torch
import torch.nn as nn
import einops
from typing import *
from healnet.baselines.multimodn.encoders import MultiModEncoder
from healnet.baselines.multimodn.decoders import MultiModDecoder
from healnet.baselines.multimodn.utils import TrainableInitState

class MultiModNModule(nn.Module):
    def __init__(self,
                 state_size: int,
                 encoders: List[MultiModEncoder], # needs to be in right order of modalities in x
                 decoders: List[MultiModDecoder], # just 1 in our case
                 err_penalty: float = 1.0, # from main pipeline
                 state_change_penalty: float = 0.0,
                 ):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_size = state_size
        # self.state = TrainableInitState(state_size=state_size, device=self.device)
        self.state = nn.Parameter(torch.randn(state_size), requires_grad=True)
        self.encoders = encoders
        self.decoders = decoders
        self.err_penalty = err_penalty
        self.state_change_penalty = state_change_penalty


        self.model = nn.ModuleList(encoders)
        self.decoder = nn.ModuleList(decoders)

    def forward(self, x: List[torch.Tensor], target: torch.Tensor) -> torch.Tensor:
        assert len(x) == len(self.encoders), "Number of inputs must match number of encoders"

        b, *_ = x[0].shape # get batch dims

        # each sample in batch gets assigned state
        # only expand once
        # if len(self.state.shape) == 1:
        self.state = nn.Parameter(einops.repeat(self.state, "d -> b d", b=b))


        running_loss = 0
        for encoder, mod in zip(self.encoders, x):
            old_state = self.state.clone()
            self.state = encoder(state=self.state, x=mod) # (l_d)

        # iterate through decoders as it's multitask
            for decoder in self.decoders:
                pred = decoder(self.state)
                loss = self.calc_loss(pred, target, old_state, self.state)
            running_loss += loss

        running_loss /= len(self.encoders)
        # reduce state over batch dim
        self.state = nn.Parameter(self.state.mean(dim=0))
        # return expected loss over batches and encoders and predictions after the last state (encoder)
        return running_loss, pred



    def calc_loss(self, pred: torch.Tensor, actual: torch.Tensor, s_old: torch.Tensor, s_new: torch.Tensor):
        b, *_ = pred.shape
        err_loss = nn.CrossEntropyLoss()(pred, actual.float())
        state_change_loss = torch.mean((s_new - s_old) ** 2)

        # mean over mini-batch
        loss = torch.mean((err_loss * self.err_penalty + state_change_loss * self.state_change_penalty), dim=0)

        return loss


