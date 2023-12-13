import torch
import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F



def nll_loss(hazards, S, Y, c, weights=None, alpha=0.4, eps=1e-7):
    """
    hazards: (n_batches, n_classes)
        The neural network output discrete survival predictions such that hazards = sigmoid(h).
    Y: (n_batches, 1)
        The true time bin index label.
    c: (n_batches, 1)
        The censoring status indicator.
    alpha: float
    eps: float
        Numerical constant; lower bound to avoid taking logs of tiny numbers.
    reduction: str
        Do we sum or average the loss function over the batches. Must be one of ['mean', 'sum']
    """
    batch_size = len(Y)
    Y = Y.view(batch_size, 1) # ground truth bin, 1,2,...,k
    c = c.view(batch_size, 1).float() #censorship status, 0 or 1
    if S is None:
        S = torch.cumprod(1 - hazards, dim=1) # surival is cumulative product of 1 - hazards
    S_padded = torch.cat([torch.ones_like(c), S], 1) #S(-1) = 0, all patients are alive from (-inf, 0) by definition
    # after padding, S(0) = S[1], S(1) = S[2], etc, h(0) = h[0]
    uncensored_loss = -(1 - c) * (torch.log(torch.gather(S_padded, 1, Y).clamp(min=eps)) + torch.log(torch.gather(hazards, 1, Y).clamp(min=eps)))
    censored_loss = - c * torch.log(torch.gather(S_padded, 1, Y+1).clamp(min=eps))
    neg_l = censored_loss + uncensored_loss
    if weights is not None:
        # Normalize weights and ensure it's a 2D tensor with the same number of rows as the input
        weights = weights / torch.sum(weights)
        weights = weights.view(1, -1).expand_as(hazards)
        # Use gather to select the weights corresponding to the target classes
        gathered_weights = torch.gather(weights, 1, Y)
        neg_l *= gathered_weights

    loss = (1-alpha) * neg_l + alpha * uncensored_loss
    loss = loss.mean()
    return loss

def nll_loss_alternative(h, y, c, alpha=0.0, eps=1e-7, reduction='mean'):
    """
    The negative log-likelihood loss function for the discrete time to event model (Zadeh and Schmid, 2020).
    Parameters
    ----------
    h: (n_batches, n_classes)
        The neural network output discrete survival predictions such that hazards = sigmoid(h).
    y: (n_batches, 1)
        The true time bin index label.
    c: (n_batches, 1)
        The censoring status indicator.
    alpha: float
    eps: float
        Numerical constant; lower bound to avoid taking logs of tiny numbers.
    reduction: str
        Do we sum or average the loss function over the batches. Must be one of ['mean', 'sum']
    """
    # print("h shape", h.shape)

    # make sure these are ints
    y = y.type(torch.int64)
    c = c.type(torch.int64)

    hazards = torch.sigmoid(h)

    S = torch.cumprod(1 - hazards, dim=1)

    S_padded = torch.cat([torch.ones_like(c), S], 1)

    s_prev = torch.gather(S_padded, dim=1, index=y).clamp(min=eps)
    h_this = torch.gather(hazards, dim=1, index=y).clamp(min=eps)
    s_this = torch.gather(S_padded, dim=1, index=y+1).clamp(min=eps)

    uncensored_loss = -(1 - c) * (torch.log(s_prev) + torch.log(h_this))
    censored_loss = - c * torch.log(s_this)


    neg_l = censored_loss + uncensored_loss
    if alpha is not None:
        loss = (1 - alpha) * neg_l + alpha * uncensored_loss

    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()
    else:
        raise ValueError("Bad input for reduction: {}".format(reduction))

    return loss


class CrossEntropySurvLoss(object):
    def __init__(self, alpha=0.15):
        self.alpha = alpha

    def __call__(self, hazards, survival, y_disc, censorship, alpha=None):
        if alpha is None:
            return ce_loss(hazards, survival, y_disc, censorship, alpha=self.alpha)
        else:
            return ce_loss(hazards, survival, y_disc, censorship, alpha=alpha)

def ce_loss(hazards, survival, y_disc, c, alpha=0.4, eps=1e-7):
    """

    Args:
        hazards:
        survival (torch.Tensor): Survival
        y_disc (torch.Tensor): ground truth bin (y_disc)
        c (torch.Tensor): censorship status indicator
        alpha:
        eps:

    Returns:

    """
    batch_size = len(y_disc)
    y_disc = y_disc.view(batch_size, 1) # ground truth bin, 1,2,...,k
    c = c.view(batch_size, 1).float() #censorship status, 0 or 1
    if survival is None:
        survival = torch.cumprod(1 - hazards, dim=1) # surival is cumulative product of 1 - hazards
    S_padded = torch.cat([torch.ones_like(c), survival], 1)
    reg = -(1 - c) * (torch.log(torch.gather(S_padded, 1, y_disc) + eps) + torch.log(torch.gather(hazards, 1, y_disc).clamp(min=eps)))
    ce_l = - c * torch.log(torch.gather(survival, 1, y_disc).clamp(min=eps)) - (1 - c) * torch.log(1 - torch.gather(survival, 1, y_disc).clamp(min=eps))
    loss = (1-alpha) * ce_l + alpha * reg
    loss = loss.mean()
    return loss


class CoxPHSurvLoss(nn.Module):

    """
    CoxPHSurvLoss: Cox Proportional Hazards Loss Function. Read more about this on
    https://en.wikipedia.org/wiki/Proportional_hazards_model
    """
    def __init__(self):
        super().__init__()

    def __call__(self, hazards, survival, censorship, **kwargs):
        """

        Args:
            survival (torch.Tensor): calculated as torch.cumprod(1 - hazards, dim=1)
            censorship (torch.Tensor): censoring status indicator
            **kwargs:

        Returns:
            float: cox loss
        """
        # This calculation credit to Travers Ching https://github.com/traversc/cox-nnet
        # Cox-nnet: An artificial neural network method for prognosis prediction of high-throughput omics data
        current_batch_len = len(survival)
        R_mat = np.zeros([current_batch_len, current_batch_len], dtype=int)
        for i in range(current_batch_len):
            for j in range(current_batch_len):
                R_mat[i,j] = survival[j] >= survival[i]

        R_mat = torch.FloatTensor(R_mat).to(device)
        theta = hazards.reshape(-1)
        exp_theta = torch.exp(theta)
        loss_cox = -torch.mean((theta - torch.log(torch.sum(exp_theta*R_mat, dim=1))) * (1 - censorship))
        return loss_cox