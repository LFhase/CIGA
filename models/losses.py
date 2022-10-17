import copy
from email.policy import default
from enum import Enum
import torch
import argparse
from torch_geometric import data
from torch_geometric.data import DataLoader

import torch.nn as nn
import torch.nn.functional as F


def get_irm_loss(causal_pred, labels, batch_env_idx, criterion=F.cross_entropy):
    device = causal_pred.device
    dummy_w = torch.tensor(1.).to(device).requires_grad_()
    loss_0 = criterion(causal_pred[batch_env_idx == 0] * dummy_w, labels[batch_env_idx == 0])
    loss_1 = criterion(causal_pred[batch_env_idx == 1] * dummy_w, labels[batch_env_idx == 1])
    grad_0 = torch.autograd.grad(loss_0, dummy_w, create_graph=True)[0]
    grad_1 = torch.autograd.grad(loss_1, dummy_w, create_graph=True)[0]
    irm_loss = torch.sum(grad_0 * grad_1)

    return irm_loss


def get_contrast_loss(causal_rep, labels, norm=None, contrast_t=1.0, sampling='mul', y_pred=None):

    if norm != None:
        causal_rep = F.normalize(causal_rep)
    if sampling.lower() in ['mul', 'var']:
        # modified from https://github.com/HobbitLong/SupContrast/blob/master/losses.py#L11
        device = causal_rep.device
        mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(1).T).float().to(device)
        # compute logits
        anchor_dot_contrast = torch.div(torch.matmul(causal_rep, causal_rep.T), contrast_t)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        # tile mask: no need
        # mask = mask.repeat(anchor_count, contrast_count)
        batch_size = labels.size(0)
        anchor_count = 1
        # mask-out self-contrast cases
        logits_mask = torch.scatter(torch.ones_like(mask), 1,
                                    torch.arange(batch_size * anchor_count).view(-1, 1).to(device), 0)
        mask = mask * logits_mask
        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        # print(log_prob)
        # print(mask.sum(1))
        # compute mean of log-likelihood over positive
        is_valid = mask.sum(1) != 0
        mean_log_prob_pos = (mask * log_prob).sum(1)[is_valid] / mask.sum(1)[is_valid]
        # some classes may not be sampled by more than 2
        # mean_log_prob_pos[torch.isnan(mean_log_prob_pos)] = 0.0

        # loss
        # contrast_loss = -(args.temperature / args.base_temperature) * mean_log_prob_pos
        # contrast_loss = contrast_loss.view(anchor_count, batch_size).mean()
        contrast_loss = -mean_log_prob_pos.mean()
        if sampling.lower() == 'var':
            contrast_loss += mean_log_prob_pos.var()
    elif sampling.lower() == 'single':
        N = causal_rep.size(0)
        pos_idx = torch.arange(N)
        neg_idx = torch.randperm(N)
        for i in range(N):
            for j in range(N):
                if labels[i] == labels[j]:
                    pos_idx[i] = j
                else:
                    neg_idx[i] = j
        contrast_loss = -torch.mean(
            torch.bmm(causal_rep.unsqueeze(1), causal_rep[pos_idx].unsqueeze(1).transpose(1, 2)) -
            torch.matmul(causal_rep.unsqueeze(1), causal_rep[neg_idx].unsqueeze(1).transpose(1, 2)))
    elif sampling.lower() == 'cncp':
        # correct & contrast with hard postive only https://arxiv.org/abs/2203.01517
        device = causal_rep.device
        mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(1).T).float().to(device)
        # compute logits
        anchor_dot_contrast = torch.div(torch.matmul(causal_rep, causal_rep.T), contrast_t)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        # tile mask: no need
        # mask = mask.repeat(anchor_count, contrast_count)
        batch_size = labels.size(0)
        anchor_count = 1
        # mask-out self-contrast cases
        logits_mask = torch.scatter(torch.ones_like(mask), 1,
                                    torch.arange(batch_size * anchor_count).view(-1, 1).to(device), 0)
        mask = mask * logits_mask
        # find hard postive & negative
        pos_mask = y_pred != labels
        neg_mask = y_pred == labels

        # hard negative: diff label && correct pred
        neg_mask = torch.logical_not(mask)  #* neg_mask
        # hard positive: same label && incorrect pred
        pos_mask = mask * pos_mask

        # compute log_prob
        neg_exp_logits = torch.exp(logits) * neg_mask
        pos_exp_logits = torch.exp(logits) * pos_mask
        log_prob = logits - \
                    torch.log(pos_exp_logits.sum(1, keepdim=True) + \
                            neg_exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        is_valid = pos_mask.sum(1) != 0
        mean_log_prob_pos = (pos_mask * log_prob).sum(1)[is_valid] / pos_mask.sum(1)[is_valid]
        # some classes may not be sampled by more than 2
        # mean_log_prob_pos[torch.isnan(mean_log_prob_pos)] = 0.0

        # loss
        # contrast_loss = -(args.temperature / args.base_temperature) * mean_log_prob_pos
        # contrast_loss = contrast_loss.view(anchor_count, batch_size).mean()
        contrast_loss = -mean_log_prob_pos.mean()
    elif sampling.lower() == 'cnc':
        # correct & contrast https://arxiv.org/abs/2203.01517
        device = causal_rep.device
        mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(1).T).float().to(device)
        # compute logits
        anchor_dot_contrast = torch.div(torch.matmul(causal_rep, causal_rep.T), contrast_t)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        # tile mask: no need
        # mask = mask.repeat(anchor_count, contrast_count)
        batch_size = labels.size(0)
        anchor_count = 1
        # mask-out self-contrast cases
        logits_mask = torch.scatter(torch.ones_like(mask), 1,
                                    torch.arange(batch_size * anchor_count).view(-1, 1).to(device), 0)
        mask = mask * logits_mask
        # find hard postive & negative
        pos_mask = y_pred != labels
        neg_mask = y_pred == labels
        # hard negative: diff label && correct pred
        neg_mask = torch.logical_not(mask) * neg_mask * logits_mask
        # hard positive: same label && incorrect pred
        pos_mask = mask * pos_mask
        if neg_mask.sum() == 0:
            neg_mask = torch.logical_not(mask)
        if pos_mask.sum() == 0:
            pos_mask = mask
        # compute log_prob
        neg_exp_logits = torch.exp(logits) * neg_mask
        pos_exp_logits = torch.exp(logits) * pos_mask
        log_prob = logits - \
                    torch.log(pos_exp_logits.sum(1, keepdim=True) + \
                            neg_exp_logits.sum(1, keepdim=True)+1e-12)
        # compute mean of log-likelihood over positive
        is_valid = pos_mask.sum(1) != 0
        mean_log_prob_pos = (pos_mask * log_prob).sum(1)[is_valid] / pos_mask.sum(1)[is_valid]
        contrast_loss = -mean_log_prob_pos.mean()
    else:
        raise Exception("Not implmented contrasting method")
    return contrast_loss
