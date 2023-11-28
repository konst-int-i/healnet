"""
Baseline model and required helper and base classes from the MM prognosis
repository: https://github.com/gevaertlab/MultimodalPrognosis/tree/master
We use this as the state-of-the-art baseline for intermediate fusion models, but
slightly modify the class to run within our main pipeline. Note that no hyperparameters
were changed from the original paper.
Used in paper Deep Learning with Multimodal Representation for Pancancer Prognosis Prediction
https://www.biorxiv.org/content/10.1101/577197v1
"""
import os, sys, random, yaml
from itertools import product
from tqdm import tqdm
from typing import *
import itertools
from box import Box
import numpy as np
import matplotlib as mpl
#mpl.use('Agg')
from sklearn.utils import shuffle
from torchvision import models

import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AbstractModel(nn.Module):

    def __init__(self):
        super(AbstractModel, self).__init__()
        self.compiled = False

    # Compile module and assign optimizer + params
    def compile(self, optimizer=None, **kwargs):

        if optimizer is not None:
            self.optimizer_class = optimizer
            self.optimizer_kwargs = kwargs

        self.optimizer = self.optimizer_class(self.parameters(), **self.optimizer_kwargs)
        self.compiled = True
        self.to(DEVICE)

    # Process a batch of data from a generator to get tensors + masks
    def __process_batch(self, data):

        data, mask = stack(data), {}
        for key in data:
            mask[key] = [int(x is not None) for x in data[key]]
            template = next((x for x in data[key] if x is not None))
            data[key] = [x if x is not None else torch.zeros_like(template) \
                            for x in data[key]]
            data[key] = torch.stack(data[key]).to(DEVICE)
            mask[key] = torch.tensor(mask[key]).to(DEVICE)

        return data, mask

    # Predict scores from a batch of data
    def predict_on_batch(self, data):

        self.eval()
        with torch.no_grad():
            data, mask = self.__process_batch(data)
            pred = self.forward(data, mask)
            pred = {key: pred[key].cpu().data.numpy() for key in pred}
            return pred

    # Fit (make one optimizer step) on a batch of data
    def fit_on_batch(self, data, target):

        self.train()
        self.zero_grad()
        self.optimizer.zero_grad()

        data, mask = self.__process_batch(data)
        pred = self.forward(data, mask)
        target = stack(target)
        target = {key: torch.stack(target[key]).to(DEVICE) for key in target}

        loss = self.loss(pred, target)
        loss.backward()
        self.optimizer.step()

        pred = {key: pred[key].cpu().data.numpy() for key in pred}
        return pred, float(loss)

    # Subclasses: please override for custom loss + forward functions
    def loss(self, pred, target):
        raise NotImplementedError()

    def forward(self, data, mask):
        raise NotImplementedError()

class TrainableModel(AbstractModel):

    def __init__(self):
        super(AbstractModel, self).__init__()
        self.compiled = False
        self.losses = []

    # Predict on generator for one epoch
    def predict(self, data, verbose=False):

        self.eval()
        with torch.no_grad():
            iterator = tqdm(data) if verbose else data
            pred = [self.predict_on_batch(batch) for batch in iterator]
            pred = np.hstack(pred)
        return pred

    # Fit on generator for one epoch
    def fit(self, datagen, validation=None, verbose=True):

        self.train()

        target = []
        iterator = tqdm(datagen) if verbose else datagen
        pred = []

        for batch, y in iterator:

            y_pred, loss = self.fit_on_batch(batch, y)
            self.losses.append(loss)

            target.append(stack(y))
            pred.append(y_pred)

            if verbose and len(self.losses) % 16 == 0:
                iterator.set_description(f"Loss: {np.mean(self.losses[-32:]):0.3f}")
                #plt.plot(self.losses)
                #plt.savefig(f"{OUTPUT_DIR}/loss.jpg");

        if verbose:
            pred, target = stack(pred), stack(target)
            pred = {key: np.concatenate(pred[key], axis=0) for key in pred}
            target = {key: np.concatenate(target[key], axis=0) for key in target}

            print (f"(training) {self.evaluate(pred, target)}")

            if validation != None:
                val_data, val_target = zip(*list(validation))
                val_pred = self.predict(val_data)

                val_target = [stack(x) for x in val_target]
                val_pred, val_target = stack(val_pred), stack(val_target)
                val_pred = {key: np.concatenate(val_pred[key], axis=0) for key in val_pred}
                val_target = {key: np.concatenate(val_target[key], axis=0) for key in val_target}

                print (f"(validation) {self.evaluate(val_pred, val_target)}")
            return self.score(val_pred, val_target)

    # Evaluate predictions and targets
    def evaluate(self, pred, target):

        scores = self.score(pred, target)
        base_scores = self.score(pred, {key: shuffle(target[key]) for key in target})

        display = []
        for key in scores:
            display.append(f"{key}={scores[key]:.4f}/{base_scores[key]:.4f}")
        return ", ".join(display)

    def eval_data(self, datagen):

        val_data, val_target = zip(*list(datagen))
        val_pred = self.predict(val_data)

        val_target = [stack(x) for x in val_target]
        val_pred, val_target = stack(val_pred), stack(val_target)
        val_pred = {key: np.concatenate(val_pred[key], axis=0) for key in val_pred}
        val_target = {key: np.concatenate(val_target[key], axis=0) for key in val_target}

        return self.score(val_pred, val_target)["C-index"]




    # Score generator based on predictions and targets
    def score(self, pred, targets):
        return NotImplementedError()

    def load(self, file_path):
        self.load_state_dict(torch.load(file_path))

    def save(self, file_path):
        torch.save(self.state_dict(), file_path)

def stack(batch, targets=None):
    """
    Turns an array (batch) of dictionaries into a dictionary of arrays
    """
    keys = batch[0].keys()
    data = {key: [] for key in keys}

    for item, key in product(batch, keys):
        data[key].append(item.get(key, None))
    return data

def masked_mean(data, masks):

    num = sum((X*mask[:, None].float() for X, mask in zip(data, masks)))
    denom = sum((mask for mask in masks))[:, None].float()
    return num/denom

def unmasked_mean(data):
    stacked_data = torch.stack(data, dim=0)
    return torch.mean(stacked_data, dim=0)

def masked_variance(data, masks):
    EX2 = masked_mean(data, masks)**2
    E2X = masked_mean((x**2 for x in data), masks)
    return E2X - EX2


class MMPrognosis(TrainableModel):
    def __init__(self,
                 output_dims: int,
                 # input_dim: int,
                 sources: List[str],
                 config: Box,
                 final_classifier_head: bool = True):
        super(MMPrognosis, self).__init__()
        self.embedding_dims = 256
        self.output_dims = output_dims
        self.config = config

        self.fcm = nn.Linear(1881, self.embedding_dims)
        self.fcc = nn.Linear(7, self.embedding_dims)
        # self.fcg = nn.Linear(60483, embedding_dims)
        self.highway = Highway(256, 10, f=F.relu)
        self.fc2 = nn.Linear(self.embedding_dims, self.output_dims)
        self.fcd = nn.Linear(self.embedding_dims, self.output_dims)
        self.bn1 = nn.BatchNorm1d(self.embedding_dims)
        self.bn2 = nn.BatchNorm1d(self.embedding_dims)
        # self.bn3 = nn.BatchNorm1d(1, affine=True)
        self.modalities = len(sources)
        self.sources = sources
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.squeezenet = models.squeezenet1_0()

        self.to_logits = nn.Sequential(
            # Reduce('b n d -> b d', 'mean'),
            nn.LayerNorm(self.embedding_dims),
            nn.Linear(self.embedding_dims, self.output_dims)
        ) if final_classifier_head else nn.Identity()


    def forward(self, data: List[torch.Tensor]):

        mask = {}

        if self.sources == ["omic"]:
            z, mask = self.omic_encoder(data[0], mask)

            mean = masked_mean((z, 1), mask["omic"])
            # mean = uasked_mean(z)
            var = masked_variance((z), mask["omic"]).mean()
            var2 = masked_mean (((z - mean.mean())**2), mask["omic"])

        elif self.sources == ["slides"]:
            w, mask = self.wsi_encoder(data[0], mask)
            mean = masked_mean((w, 1), mask["slides"])
            var = masked_variance((w), mask["slides"]).mean()
            var2 = masked_mean (((w - mean.mean())**2), mask["slides"])

        elif self.sources == ["omic", "slides"]:
            z, mask = self.omic_encoder(data[0], mask)
            w, mask = self.wsi_encoder(data[1], mask)
            mean = masked_mean((z, w), (mask["omic"], mask["slides"]))

            var = masked_variance((z, w), (mask["omic"], mask["slides"])).mean()
            var2 = masked_mean (( (z - mean.mean())**2, (w - mean.mean())**2),\
                                (mask["omic"], mask["slides"]))


        ratios = var/var2.mean(dim=1)
        ratio = ratios.clamp(min=0.02, max=1.0).mean()

        x = mean

        if self.config["train_loop.batch_size"] > 1:
            x = self.bn1(x)
        x = F.dropout(x, 0.5, training=self.training)
        x = self.highway(x)
        if self.config["train_loop.batch_size"] > 1:
            x = self.bn2(x)

        # TODO: convert to logits
        logits = self.to_logits(x)

        return logits

        # score = F.log_softmax(self.fc2(x), dim=1)
        # hazard = self.fcd(x)
        #
        # return {"score": score, "hazard": hazard, "ratio": ratio.unsqueeze(0)}

    def wsi_encoder(self, data, mask):
        w = data
        input_dim = w.shape[1]
        # convolution across patches
        conv1 = nn.Conv1d(in_channels=input_dim, out_channels=512, kernel_size=5, stride=2, padding=2,
                          device=self.device)
        conv2 = nn.Conv1d(in_channels=512, out_channels=self.embedding_dims, kernel_size=5, stride=2, padding=2,
                          device=self.device)
        global_avg_pool = nn.AdaptiveAvgPool1d(1)
        # encoder instead of sqeeze net
        w = conv1(w)
        w = F.relu(w)
        w = conv2(w)
        w = F.relu(w)
        w = global_avg_pool(w)
        w = w.squeeze(-1)
        mask["slides"] = torch.ones(w.shape[1], 1).to(self.device)
        return w, mask

    def omic_encoder(self, data, mask):
        z = data
        z = z.view(z.shape[0], -1)
        self.fcg = nn.Linear(in_features=z.shape[1], out_features=self.embedding_dims, device=self.device)
        z = torch.tanh(self.fcg(z))
        mask["omic"] = torch.ones(z.shape[1], 1).to(self.device)
        return z, mask

    def loss(self, pred, target):

        vital_status = target["vital_status"]
        days_to_death = target["days_to_death"]
        hazard = pred["hazard"].squeeze()

        loss = F.nll_loss(pred["score"], vital_status)

        _, idx = torch.sort(days_to_death)
        hazard_probs = F.softmax(hazard[idx].squeeze()[1-vital_status.byte()])
        hazard_cum = torch.stack([torch.tensor(0.0)] + list(accumulate(hazard_probs)))
        N = hazard_probs.shape[0]
        weights_cum = torch.range(1, N)
        p, q = hazard_cum[1:], 1-hazard_cum[:-1]
        w1, w2 = weights_cum, N - weights_cum

        probs = torch.stack([p, q], dim=1)
        logits = torch.log(probs)
        ll1 = (F.nll_loss(logits, torch.zeros(N).long(), reduce=False) * w1)/N
        ll2 = (F.nll_loss(logits, torch.ones(N).long(), reduce=False) * w2)/N
        loss2 = torch.mean(ll1 + ll2)

        loss3 = pred["ratio"].mean()

        return loss + loss2 + loss3*0.3

    # def score(self, pred, target):
    #     vital_status = target["vital_status"]
    #     days_to_death = target["days_to_death"]
    #     score_pred = pred["score"][:, 1]
    #     hazard = pred["hazard"][:, 0]
    #
    #     auc = roc_auc_score(vital_status, score_pred)
    #     cscore = concordance_index(days_to_death, -hazard, np.logical_not(vital_status))
    #
    #     return {"AUC": auc, "C-index": cscore, "Ratio": pred["ratio"].mean()}





class Highway(nn.Module):

    def __init__(self, size, num_layers, f):

        super(Highway, self).__init__()

        self.num_layers = num_layers
        self.nonlinear = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])
        self.linear = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])
        self.gate = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])
        self.f = f

    def forward(self, x):

        for layer in range(self.num_layers):
            gate = torch.sigmoid(self.gate[layer](x))
            nonlinear = self.f(self.nonlinear[layer](x))
            linear = self.linear[layer](x)
            x = gate * nonlinear + (1 - gate) * linear

        return x




