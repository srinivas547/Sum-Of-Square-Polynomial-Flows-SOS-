import copy
import math
import sys
import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

import datasets
from models import SOSFlow, MADE, FlowSequential, BatchNormFlow, Reverse

MODEL_DIR = "trained_models/"


#
#   Model utilities
#

def build_model(input_size, hidden_size, k, r, n_blocks, lr, device=None, **kwargs):
    modules = []
    for i in range(n_blocks):
        modules += [
            SOSFlow(input_size, hidden_size, k, r),
            BatchNormFlow(input_size),
            Reverse(input_size)
        ]
    model = FlowSequential(*modules)
    if device is not None:
        model.to(device)

    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    return model, optimizer

def build_maf(num_inputs, num_hidden, num_blocks, lr, device, act='relu'):
    modules = []

    for _ in range(num_blocks):
        modules += [
            MADE(num_inputs, num_hidden, act=act),
            BatchNormFlow(num_inputs),
            Reverse(num_inputs)
        ]

    model = FlowSequential(*modules)

    '''
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight)
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data.fill_(0)
    '''

    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)

    return model, optimizer

def flow_loss(z, logdet, size_average=True, use_cuda=True):
    # If using Student-t as source distribution#
    #df = torch.tensor(5.0)
    #if use_cuda:
     #   log_prob = log_prob_st(z, torch.tensor([5.0]).cuda())
    #else:
        #log_prob = log_prob_st(z, torch.tensor([5.0]))
    #log_probs = log_prob.sum(-1, keepdim=True)
    ''' If using Uniform as source distribution
    log_probs = 0
    '''
    log_probs = (-0.5 * z.pow(2) - 0.5 * np.log(2 * np.pi)).sum(-1, keepdim=True)
    loss = -(log_probs + logdet).sum()
    # CHANGED TO UNIFORM SOURCE DISTRIBUTION
    #loss = -(logdet).sum()
    if size_average:
        loss /= z.size(0)
    return loss


#
#   Loading Data
#


def make_datasets(train_data, val_data=None, test_data=None):
    train_dataset = _make_dataset(train_data)
    valid_dataset = _make_dataset(val_data) if val_data is not None else None
    test_dataset = _make_dataset(test_data) if test_data is not None else None
    return train_dataset, valid_dataset, test_dataset

def _make_dataset(all_data):
    if type(all_data) == list:
        data_tensors = [torch.from_numpy(data).float() for data in all_data]
        dataset = torch.utils.data.TensorDataset(*data_tensors)
    else:
        data_tensor = torch.from_numpy(all_data).float()
        dataset = torch.utils.data.TensorDataset(data_tensor)
    return dataset

def make_loaders(train_dataset, valid_dataset, test_dataset, batch_size, test_batch_size, **kwargs):
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, **kwargs)

    valid_loader = None if valid_dataset is None else torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        drop_last=False,
        **kwargs)

    test_loader = None if test_dataset is None else torch.utils.data.DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        drop_last=False,
        **kwargs)

    return train_loader, valid_loader, test_loader

def load_POWER(batch_size, test_batch_size):
    dataset = getattr(datasets, "POWER")()
    train_dataset, valid_dataset, test_dataset = make_datasets(dataset.trn.x, dataset.val.x, dataset.tst.x)
    train_loader, valid_loader, test_loader = make_loaders(train_dataset, valid_dataset, test_dataset,
                                                           batch_size, test_batch_size)
    return train_loader, valid_loader, test_loader

#
#   Utilities
#

def get_batch(loader, device=None):
    batch = next(iter(loader))[0]
    return batch.to(device) if device is not None else batch

def plot_hist(model, batch):
    import matplotlib.pyplot as plt
    zhat, logdet = model(batch)
    for i in range(zhat.size(1)):
        plt.hist(zhat.detach()[:,i])
    plt.show()

#
#   Training functions
#

def _eval(model, batch, device, size_average=True):
    if isinstance(batch, list):
        batch = batch[0]
    batch = batch.to(device)
    zhat, log_jacob = model(batch)
    loss = flow_loss(zhat, log_jacob, size_average=size_average)
    return loss


def train_epoch(model, optim, train_loader, epoch, device, log_interval):
    model.train()
    epoch_loss = 0
    for batch_idx, data in enumerate(train_loader):
        data_size = len(data[0]) if isinstance(data, list) else len(data)
        optim.zero_grad()
        loss = _eval(model, data, device)
        loss.backward()
        optim.step()
        epoch_loss += loss.item()
        if (batch_idx+1) % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (batch_idx+1) * data_size, len(train_loader.dataset),
                100. * (batch_idx+1) / len(train_loader), epoch_loss / (log_interval)))
            epoch_loss=0

    for module in model.modules():
        if isinstance(module, BatchNormFlow):
            module.momentum = 0
    
    with torch.no_grad():
        model(train_loader.dataset.tensors[0].to(device))
    
    for module in model.modules():
        if isinstance(module, BatchNormFlow):
            module.momentum = 1


def validate(model, loader, device, prefix='Validation'):
    model.eval()
    val_loss = 0

    for data in loader:
        with torch.no_grad():
            val_loss += _eval(model, data, device, size_average=True).item()

    val_loss /= len(loader)
    print('\n{} set: Average loss: {:.4f}\n'.format(prefix, val_loss))

    return val_loss


def train(model, optim, train_loader, valid_loader, test_loader, epochs, device, log_interval):
    best_validation_loss = float('inf')
    best_validation_epoch = 0
    best_model = model

    for epoch in range(epochs):
        train_epoch(model, optim, train_loader, epoch, device, log_interval)
        if valid_loader is not None:
            validation_loss = validate(model, valid_loader, device)

            if epoch - best_validation_epoch >= 30:
                break

            if validation_loss < best_validation_loss:
                best_validation_epoch = epoch
                best_validation_loss = validation_loss
                #best_model = copy.deepcopy(model)

            print('Best validation at epoch {}: Average loss: {:.4f}\n'.format(
                best_validation_epoch, best_validation_loss))

    if valid_loader is not None and test_loader is not None:
        test_loss = validate(best_model, test_loader, device, prefix='Test')
    else:
        best_model = model
        test_loss = -1
    return best_model, test_loss


#
#   Loading Models
#

def default_name(t=None, prefix="model_", suffix=".pt"):
    if t is None:
        t = datetime.datetime.now()
    timestamp = str(t.day) + "_" + str(t.hour) + "_" + str(t.minute)
    return prefix + timestamp + suffix


def load_model(name, cpu=True):
    dict = torch.load(name, map_location='cpu') if cpu else torch.load(MODEL_DIR + name)
    model = dict['model']
    #optim = dict['optim']
    args = dict['args']
    return model, args

def log_prob_st(z, df):
    #if self._validate_args:
    #   self._validate_sample(value)
    #y = (value - self.loc) / self.scale
    Z = (0.5 * torch.log(df) + 0.5 * math.log(math.pi) + torch.lgamma(0.5 * df) - torch.lgamma(0.5 * (df + 1.)))
    return -0.5 * (df + 1.) * torch.log1p(z.pow(2) / df) - Z