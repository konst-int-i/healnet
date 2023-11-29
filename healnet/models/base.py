"""
Some abstract base classes required to run baselines
"""

import os, sys, random, yaml
from itertools import product
from tqdm import tqdm

import numpy as np
import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import optim

# from utils import *
import IPython

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

def masked_variance(data, masks):
    EX2 = masked_mean(data, masks)**2
    E2X = masked_mean((x**2 for x in data), masks)
    return E2X - EX2