
from __future__ import absolute_import, division, print_function

import torch
from Utils import common as cmn, data_gen
from Utils.common import count_correct
from Models.stochastic_layers import StochasticLayer
from Utils.Losses import get_loss_func



# -----------------------------------------------------------------------------------------------------------#

def run_eval_Bayes(model, loader, prm, verbose=0):

    with torch.no_grad():    # no need for backprop in test

        if len(loader) == 0:
            return 0.0, 0.0
        if prm.test_type == 'Expected':
            info = run_eval_expected(model, loader, prm)
        elif prm.test_type == 'MaxPosterior':
            info = run_eval_max_posterior(model, loader, prm)
        elif prm.test_type == 'MajorityVote':
            info = run_eval_majority_vote(model, loader, prm, n_votes=5)
        elif prm.test_type == 'AvgVote':
            info = run_eval_avg_vote(model, loader, prm, n_votes=5)
        else:
            raise ValueError('Invalid test_type')
        if verbose:
            print('Accuracy: {:.3} ({}/{}), loss: {:.4}'.format(float(info['test_acc']), info['n_correct'],
                                                                          info['n_samples'], float(info['avg_loss'])))
    return info['acc'], info['avg_loss']
# -------------------------------------------------------------------------------------------

def run_eval_max_posterior(model, loader, prm):
    ''' Estimates the the loss by using the mean network parameters'''
    n_samples = len(loader.dataset)
    loss_criterion = get_loss_func(prm)
    model.eval()
    avg_loss = 0
    n_correct = 0
    for batch_data in loader:
        inputs, targets = data_gen.get_batch_vars(batch_data, prm)
        batch_size = inputs.shape[0]
        old_eps_std = model.set_eps_std(0.0)   # test with max-posterior
        outputs = model(inputs)
        model.set_eps_std(old_eps_std)  # return model to normal behaviour
        avg_loss += loss_criterion(outputs, targets).item() # sum the loss contributed from batch
        n_correct += count_correct(outputs, targets)

    avg_loss /= n_samples
    acc = n_correct / n_samples
    info = {'acc':acc, 'n_correct':n_correct,
            'n_samples':n_samples, 'avg_loss':avg_loss}
    return info


# -------------------------------------------------------------------------------------------

def run_eval_expected(model, loader, prm):
    ''' Estimates the expectation of the loss by monte-carlo averaging'''
    n_samples = len(loader.dataset)
    loss_criterion = get_loss_func(prm)
    model.eval()
    avg_loss = 0.0
    n_correct = 0
    n_MC = prm.n_MC_eval # number of monte-carlo runs for expected loss estimation
    for batch_data in loader:
        inputs, targets = data_gen.get_batch_vars(batch_data, prm)
        batch_size = inputs.shape[0]
        #  monte-carlo runs
        for i_MC in range(n_MC):
            outputs = model(inputs)
            avg_loss += loss_criterion(outputs, targets).item() # sum the loss contributed from batch
            n_correct += count_correct(outputs, targets)

    avg_loss /= (n_MC * n_samples)
    acc = n_correct / (n_MC * n_samples)
    info = {'acc':acc, 'n_correct':n_correct,
            'n_samples':n_samples, 'avg_loss':avg_loss}
    return info

# -------------------------------------------------------------------------------------------
def run_eval_majority_vote(model, loader, prm, n_votes=5):
    ''' Estimates the the loss of the the majority votes over several draws form network's distribution'''

    loss_criterion = get_loss_func(prm)
    n_samples = len(loader.dataset)
    n_test_batches = len(loader)
    model.eval()
    avg_loss = 0
    n_correct = 0
    for batch_data in loader:
        inputs, targets = data_gen.get_batch_vars(batch_data, prm)

        batch_size = inputs.shape[0] # min(prm.test_batch_size, n_samples)
        info = data_gen.get_info(prm)
        n_labels = info['n_classes']
        votes = torch.zeros((batch_size, n_labels), device=prm.device)
        loss_from_batch = 0.0
        for i_vote in range(n_votes):

            outputs = model(inputs)
            loss_from_batch += loss_criterion(outputs, targets).item()
            pred = outputs.data.max(1, keepdim=True)[1]  # get the index of the max output
            for i_sample in range(batch_size):
                pred_val = pred[i_sample].cpu().numpy()[0]
                votes[i_sample, pred_val] += 1
        avg_loss += loss_from_batch / n_votes # sum the loss contributed from batch

        majority_pred = votes.max(1, keepdim=True)[1] # find argmax class for each sample
        n_correct += majority_pred.eq(targets.data.view_as(majority_pred)).cpu().sum()
    avg_loss /= n_samples
    acc = n_correct / n_samples
    info = {'acc': acc, 'n_correct': n_correct,
            'n_samples': n_samples, 'avg_loss': avg_loss}
    return info
# -------------------------------------------------------------------------------------------

def run_eval_avg_vote(model, loader, prm, n_votes=5):
    ''' Estimates the the loss by of the average vote over several draws form network's distribution'''

    loss_criterion = get_loss_func(prm)
    n_samples = len(loader.dataset)
    n_test_batches = len(loader)
    model.eval()
    avg_loss = 0
    n_correct = 0
    for batch_data in loader:
        inputs, targets = data_gen.get_batch_vars(batch_data, prm)

        batch_size = min(prm.test_batch_size, n_samples)
        info = data_gen.get_info(prm)
        n_labels = info['n_classes']
        votes = torch.zeros((batch_size, n_labels), device=prm.device)
        loss_from_batch = 0.0
        for i_vote in range(n_votes):

            outputs = model(inputs)
            loss_from_batch += loss_criterion(outputs, targets).item()
            votes += outputs.data

        majority_pred = votes.max(1, keepdim=True)[1]
        n_correct += majority_pred.eq(targets.data.view_as(majority_pred)).cpu().sum()
        avg_loss += loss_from_batch / n_votes  # sum the loss contributed from batch

    avg_loss /= n_samples
    acc = n_correct / n_samples
    info = {'acc': acc, 'n_correct': n_correct,
            'n_samples': n_samples, 'avg_loss': avg_loss}
    return info
# -------------------------------------------------------------------------------------------


# -------------------------------------------------------------------------------------------

def set_model_values(model, mean, log_var):
    layers_list = [layer for layer in model.children() if isinstance(layer, StochasticLayer)]

    for i_layer, layer in enumerate(layers_list):
        if hasattr(layer, 'w'):
            init_param(layer.w['log_var'], log_var)
            init_param(layer.w['mean'], mean)
        if hasattr(layer, 'b'):
            init_param(layer.b['log_var'], log_var)
            init_param(layer.b['mean'], mean)


def init_param(x, init_val):
    if isinstance(init_val, dict):
        # In case of a random init:
        x.data.normal_(init_val['mean'], init_val['std'])
    else:
        # In case of a fixed init:
        x.data.fill_(init_val)

# -------------------------------------------------------------------------------------------