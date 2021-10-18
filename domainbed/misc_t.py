import os
from domainbed.lib import misc
import numpy as np
import torch
import torch.nn.functional as F
import json

def argmax(iterable):
    return max(enumerate(iterable), key=lambda x: x[1])[0]

def argmin(iterable):
    return min(enumerate(iterable), key=lambda x: x[1])[0]

def percent(number):
    number = round(number, 4) * 100
    return f"{number:.2f}\\%"

def percent_no(number):
    number = round(number, 4) * 100
    return f"{number:.2f}"

def round_l(num_list, digit=4):
    return [round(num, digit) for num in num_list] 

def loss_gap(num_domains, env_loaders, model, device, whole=False):
    ''' compute gap = max_i loss_i(h) - min_j loss_j(h), return i, j, and the gap for the whole dataset'''
    ''' model = h, index are from 1 ... n'''
    max_env_loss, min_env_loss = -np.inf, np.inf
    max_index = min_index = 0
    index = 0
    for index, loader in enumerate(env_loaders):
        loss = misc.loss(algorithm, loader, device)
        # print("index: ", index, "loss: ", loss)
        if index == 0 and not whole:
            continue
        if loss > max_env_loss:
            max_env_loss = loss
            max_index = index
        if loss < min_env_loss:
            min_env_loss = loss
            min_index = index
    return max_index, min_index, max_env_loss, min_env_loss


def loss_gap_batch(num_domains, minibatches, model, device):
    ''' compute gap = max_i loss_i(h) - min_j loss_j(h), return i, j, and the gap for a single batch'''
    ''' because we will compute sup_{h'} (rather than min), the return value is the opposite'''
    ''' model = h, index are from 1 ... n, minibatches is a list of iterators for a given batch'''
    max_env_loss, min_env_loss = -np.inf, np.inf
    for index, (x, y) in enumerate(minibatches):
        x = x.to(device)
        y = y.to(device)
        p = model.predict(x)
        batch = torch.ones(len(x)).to(device)
        total = batch.sum().item()
        loss = F.cross_entropy(p, y) * len(y) / total
        if loss > max_env_loss:
            max_env_loss = loss
        if loss < min_env_loss:
            min_env_loss = loss
    return min_env_loss - max_env_loss


def distance(h1, h2):
    ''' distance of two networks (h1, h2 are classifiers)'''
    dist = 0.
    for param in h1.state_dict():
        h1_param, h2_param = h1.state_dict()[param], h2.state_dict()[param]
        dist += torch.norm(h1_param - h2_param) ** 2  # use Frobenius norms for matrices
    return torch.sqrt(dist)


def proj(delta, adv_h, h):
    ''' return proj_{B(h, \delta)}(adv_h), Euclidean projection to Euclidean ball'''
    ''' adv_h and h are two classifiers'''
    dist = distance(adv_h, h)
    if dist <= delta:
        return adv_h
    else:
        ratio = delta / dist
        for param_h, param_adv_h in zip(h.parameters(), adv_h.parameters()):
            param_adv_h.data = param_h + ratio * (param_adv_h - param_h)
        # print("distance: ", distance(adv_h, h))
        return adv_h

def loss_acc(num_domains, env_loaders, model, device):
    '''evaluate a tuple of losses (of each domain) and a tuple of accs (of each domain)'''
    losses, accs = [], []
    for i in range(num_domains):
        print(f'domain {i}')
        env = env_loaders[i]
        loss = misc.loss(model, env, device)
        acc = misc.accuracy(model, env, None, device)
        losses.append(loss)
        accs.append(acc)
    return losses, accs
