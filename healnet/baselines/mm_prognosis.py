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

import itertools
import numpy as np
import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms, models
from sklearn.metrics import accuracy_score, roc_auc_score
from lifelines.utils import concordance_index
from scipy.stats import pearsonr

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


class MMPrognosis(TrainableModel):
    def __init__(self):
        super(MMPrognosis, self).__init__()

        self.fcm = nn.Linear(1881, 256)
        self.fcc = nn.Linear(7, 256)
        self.fcg = nn.Linear(60483, 256)
        self.highway = Highway(256, 10, f=F.relu)
        self.fc2 = nn.Linear(256, 2)
        self.fcd = nn.Linear(256, 1) # TODO: 4 out features
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(1, affine=True)

        self.squeezenet = models.squeezenet1_0()

    def forward(self, data, mask):

        x = data['mirna']
        x = x.view(x.shape[0], -1)
        x = F.dropout(x, 0.3)
        x = F.tanh(self.fcm(x))

        y = data['clinical']
        y = y.view(y.shape[0], -1)
        y = F.tanh(self.fcc(y))

        z = data['gene']
        z = z.view(z.shape[0], -1)
        z = F.tanh(self.fcg(z))

        w = data['slides']
        B, N, C, H, W = w.shape
        print ("Slides shape: ", w.shape)
        w = w.view(w.shape[0], -1)
        w = F.tanh(self.squeezenet(w.view(B*N, C, H, W)).view(B, N, -1).mean(dim=1))

        # TODO: add masks (all ones in our case, no missing mods)
        mean = masked_mean((x, y, z, w), (mask["mirna"], mask["clinical"], mask["gene"], mask["slides"]))


        var = masked_variance((x, y, z, w), (mask["mirna"], mask["clinical"], mask["gene"], mask["slides"])).mean()
        var2 = masked_mean (((x - mean.mean())**2, (y - mean.mean())**2, (z - mean.mean())**2, (w - mean.mean())**2),\
                            (mask["mirna"], mask["clinical"], mask["gene"], mask["slides"]))
        # calculate ratio of variances via expectation formula: Var(X) = E[(X-X^)^2]

        ratios = var/var2.mean(dim=1)
        ratio = ratios.clamp(min=0.02, max=1.0).mean()

        x = mean

        x = self.bn1(x)
        x = F.dropout(x, 0.5, training=self.training)
        x = self.highway(x)
        x = self.bn2(x)

        # TODO: convert to logits

        score = F.log_softmax(self.fc2(x), dim=1)
        hazard = self.fcd(x)

        return {"score": score, "hazard": hazard, "ratio": ratio.unsqueeze(0)}

    # def loss(self, pred, target):
    #
    #     vital_status = target["vital_status"]
    #     days_to_death = target["days_to_death"]
    #     hazard = pred["hazard"].squeeze()
    #
    #     loss = F.nll_loss(pred["score"], vital_status)
    #
    #     _, idx = torch.sort(days_to_death)
    #     hazard_probs = F.softmax(hazard[idx].squeeze()[1-vital_status.byte()])
    #     hazard_cum = torch.stack([torch.tensor(0.0)] + list(accumulate(hazard_probs)))
    #     N = hazard_probs.shape[0]
    #     weights_cum = torch.range(1, N)
    #     p, q = hazard_cum[1:], 1-hazard_cum[:-1]
    #     w1, w2 = weights_cum, N - weights_cum
    #
    #     probs = torch.stack([p, q], dim=1)
    #     logits = torch.log(probs)
    #     ll1 = (F.nll_loss(logits, torch.zeros(N).long(), reduce=False) * w1)/N
    #     ll2 = (F.nll_loss(logits, torch.ones(N).long(), reduce=False) * w2)/N
    #     loss2 = torch.mean(ll1 + ll2)
    #
    #     loss3 = pred["ratio"].mean()
    #
    #     return loss + loss2 + loss3*0.3

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
            gate = F.sigmoid(self.gate[layer](x))
            nonlinear = self.f(self.nonlinear[layer](x))
            linear = self.linear[layer](x)
            x = gate * nonlinear + (1 - gate) * linear

        return x


class AbstractPatientGenerator(object):
	def __init__(self, cases=fetch.cases, samples=500, val_samples=100, verbose=False):
		super(AbstractPatientGenerator, self).__init__()
		self.train_cases, self.val_cases = train_test_split(list(cases), test_size=0.15)
		self.samples, self.val_samples = samples, val_samples
		self.verbose = verbose

	def data(self, mode='train', cases=None):

		case_list = self.train_cases if mode == 'train' else self.val_cases
		num_samples = self.samples if mode == 'train' else self.val_samples

		cases = cases or (random.choice(case_list) for i in itertools.repeat(0))
		samples = (self.sample(case, mode=mode) for case in cases)
		samples = itertools.islice((x for x in samples if x is not None), num_samples)

		return samples

	def sample(self, case, mode='train'):
		raise NotImplementedError()

