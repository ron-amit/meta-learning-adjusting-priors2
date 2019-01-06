
from __future__ import absolute_import, division, print_function


import torch
from torch.autograd import Variable
import math
from Utils import common as cmn
import torch.nn.functional as F
from Models.stochastic_layers import StochasticLayer
from Utils.common import net_weights_magnitude, count_correct
# -----------------------------------------------------------------------------------------------------------#


def get_hyper_divergnce(prm, prior_model):
    ''' calculates a divergence between hyper-prior and hyper-posterior....
     which is, in our case, just a regularization term over the prior parameters  '''

    # Note:  the hyper-prior is N(0, kappa_prior^2 * I)
    # Note:  the hyper-posterior is N(parameters-of-prior-distribution, kappa_post^2 * I)

    # KLD between hyper-posterior and hyper-prior:
    hyper_dvrg = (1 / (2 * prm.kappa_prior ** 2)) * net_weights_magnitude(prior_model, prm, p=2)

    return hyper_dvrg


# -----------------------------------------------------------------------------------------------------------#

def get_meta_complexity_term(hyper_kl, prm, n_train_tasks):
    if n_train_tasks == 0:
        meta_complex_term = 0.0  # infinite tasks case
    else:
        if prm.complexity_type == 'McAllester' or  prm.complexity_type == 'Seeger':
            delta = prm.delta
            meta_complex_term = torch.sqrt(hyper_kl / (2*n_train_tasks) + math.log(4*math.sqrt(n_train_tasks) / delta))

        elif prm.complexity_type == 'PAC_Bayes_Pentina':
            meta_complex_term = hyper_kl / math.sqrt(n_train_tasks)

        elif prm.complexity_type == 'Variational_Bayes':
            meta_complex_term = hyper_kl

        elif prm.complexity_type == 'NoComplexity':
            meta_complex_term = 0.0

        else:
            raise ValueError('Invalid complexity_type')
    return meta_complex_term
# -----------------------------------------------------------------------------------------------------------#

#  -------------------------------------------------------------------------------------------
#  Intra-task complexity for posterior distribution
# -------------------------------------------------------------------------------------------
def get_task_complexity(prm, prior_model, post_model, n_samples, avg_empiric_loss, hyper_dvrg=0, n_train_tasks=1, dvrg=None, noised_prior=False):

    complexity_type = prm.complexity_type
    delta = prm.delta  #  maximal probability that the bound does not hold

    if not dvrg:
        # calculate divergence between posterior and sampled prior
        dvrg = get_net_densities_divergence(prior_model, post_model, prm, noised_prior)

    if complexity_type == 'NoComplexity':
        # set as zero
        complex_term = torch.zeros(1, requires_grad=False, device=prm.device)

    elif prm.complexity_type == 'McAllester':
        # According to 'Simplified PAC-Bayesian Margin Bounds', McAllester 2003
        # complex_term = torch.sqrt((1 / (2 * (n_samples-1))) * (hyper_dvrg + div + math.log(2 * n_samples / delta)))
        complex_term = torch.sqrt((hyper_dvrg + dvrg + math.log(2 * n_samples / delta)) / (2 * (n_samples - 1)))

    elif prm.complexity_type == 'Seeger':
        # According to 'Simplified PAC-Bayesian Margin Bounds', McAllester 2003
        seeger_eps = (dvrg + hyper_dvrg + math.log(2 * math.sqrt(n_samples) / delta)) / n_samples
        sqrt_arg = 2 * seeger_eps * avg_empiric_loss
        # sqrt_arg = F.relu(sqrt_arg)  # prevent negative values due to numerical errors
        complex_term = 2 * seeger_eps + torch.sqrt(sqrt_arg)

    elif prm.complexity_type == 'Catoni':
        # See "From PAC-Bayes Bounds to KL Regularization" Germain 2009
        # & Olivier Catoni. PAC-Bayesian surpevised classification: the thermodynamics of statistical learning
        complex_term = avg_empiric_loss + (2 / n_samples) * (hyper_dvrg + dvrg + math.log(1/ delta))


    # elif prm.complexity_type == 'Seeger2':
    #     # According to 'Simplified PAC-Bayesian Margin Bounds', McAllester 2003
    #     seeger_eps = (1 / (n_samples - 1)) * (div + hyper_dvrg + math.log(n_samples / delta))
    #     sqrt_arg = 2 * seeger_eps * avg_empiric_loss
    #     # sqrt_arg = F.relu(sqrt_arg)  # prevent negative values due to numerical errors
    #     complex_term = 2 * seeger_eps + torch.sqrt(sqrt_arg)


    elif complexity_type == 'PAC_Bayes_Pentina':
        complex_term = math.sqrt(1 / n_samples) * dvrg + hyper_dvrg * (1 / (n_train_tasks * math.sqrt(n_samples)))

    elif complexity_type == 'Variational_Bayes':
        # Since we approximate the expectation of the likelihood of all samples,
        # we need to multiply by the average empirical loss by total number of samples
        # this will be done later
        complex_term = dvrg


    else:
        raise ValueError('Invalid complexity_type')

    return complex_term
# -------------------------------------------------------------------------------------------


def get_net_densities_divergence(prior_model, post_model, prm, noised_prior=False):

    prior_layers_list = [layer for layer in prior_model.children() if isinstance(layer, StochasticLayer)]
    post_layers_list = [layer for layer in post_model.children() if isinstance(layer, StochasticLayer)]

    total_dvrg = 0
    for i_layer, prior_layer in enumerate(prior_layers_list):
        post_layer = post_layers_list[i_layer]
        if hasattr(prior_layer, 'w'):
            total_dvrg += get_dvrg_element(post_layer.w, prior_layer.w, prm, noised_prior)
        if hasattr(prior_layer, 'b'):
            total_dvrg += get_dvrg_element(post_layer.b, prior_layer.b, prm, noised_prior)


    return total_dvrg
# -------------------------------------------------------------------------------------------

def  get_dvrg_element(post, prior, prm, noised_prior=False):
    """KL divergence D_{KL}[post(x)||prior(x)] for a fully factorized Gaussian"""

    if noised_prior and prm.kappa_post > 0:
        prior_log_var = add_noise(prior['log_var'], prm.kappa_post)
        prior_mean = add_noise(prior['mean'], prm.kappa_post)
    else:
        prior_log_var = prior['log_var']
        prior_mean = prior['mean']

    post_var = torch.exp(post['log_var'])
    prior_var = torch.exp(prior_log_var)
    post_std = torch.exp(0.5 * post['log_var'])
    prior_std = torch.exp(0.5 * prior_log_var)


   # Calculate KL divergence between two Gaussian vectors:
    numerator = (post['mean'] - prior_mean).pow(2) + post_var
    denominator = prior_var
    div_elem = 0.5 * torch.sum(prior_log_var - post['log_var'] + numerator / denominator - 1)

    # note: don't add small number to denominator, since we need to have zero KL when post==prior.

    return div_elem
# -------------------------------------------------------------------------------------------

def add_noise(param, std):
    return param + Variable(param.data.new(param.size()).normal_(0, std), requires_grad=False)
# -------------------------------------------------------------------------------------------

def add_noise_to_model(model, std):

    layers_list = [layer for layer in model.children() if isinstance(layer, StochasticLayer)]

    for i_layer, layer in enumerate(layers_list):
        if hasattr(layer, 'w'):
            add_noise(layer.w['log_var'], std)
            add_noise(layer.w['mean'], std)
        if hasattr(layer, 'b'):
            add_noise(layer.b['log_var'], std)
            add_noise(layer.b['mean'], std)