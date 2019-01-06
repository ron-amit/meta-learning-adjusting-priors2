

from __future__ import absolute_import, division, print_function

import timeit
from copy import deepcopy
import torch
import numpy as np
from Models.stochastic_models import get_model
from Utils import common as cmn, data_gen
from Utils.Bayes_utils import run_eval_Bayes
from Utils.complexity_terms import get_task_complexity
from Utils.common import grad_step, correct_rate, write_to_log
from Utils.Losses import get_loss_func
import matplotlib.pyplot as plt
# -------------------------------------------------------------------------------------------
#  Stochastic Single-task learning
# -------------------------------------------------------------------------------------------

def run_learning(data_loader, prm, prior_model=None, init_from_prior=True, verbose=1):

    # -------------------------------------------------------------------------------------------
    #  Setting-up
    # -------------------------------------------------------------------------------------------

    # Unpack parameters:
    optim_func, optim_args, lr_schedule = \
        prm.optim_func, prm.optim_args, prm.lr_schedule

    # Loss criterion
    loss_criterion = get_loss_func(prm)

    train_loader = data_loader['train']
    test_loader = data_loader['test']
    n_batches = len(train_loader)
    n_train_samples = data_loader['n_train_samples']

    figure_flag = hasattr(prm, 'log_figure') and prm.log_figure

    # get model:
    if prior_model and init_from_prior:
        # init from prior model:
        post_model = deepcopy(prior_model).to(prm.device)
    else:
        post_model = get_model(prm)

    # post_model.set_eps_std(0.0) # DEBUG: turn off randomness

    #  Get optimizer:
    optimizer = optim_func(post_model.parameters(), **optim_args)

    # -------------------------------------------------------------------------------------------
    #  Training epoch  function
    # -------------------------------------------------------------------------------------------

    def run_train_epoch(i_epoch, log_mat):

        post_model.train()

        for batch_idx, batch_data in enumerate(train_loader):

            # get batch data:
            inputs, targets = data_gen.get_batch_vars(batch_data, prm)

            batch_size = inputs.shape[0]

            # Monte-Carlo iterations:
            avg_empiric_loss = torch.zeros(1, device=prm.device)
            n_MC = prm.n_MC

            for i_MC in range(n_MC):

                # calculate objective:
                outputs = post_model(inputs)
                avg_empiric_loss_curr = (1 / batch_size) * loss_criterion(outputs, targets)
                avg_empiric_loss += (1 / n_MC) * avg_empiric_loss_curr

            # complexity/prior term:
            if prior_model:
                complexity_term = get_task_complexity(
                    prm, prior_model, post_model, n_train_samples, avg_empiric_loss)
            else:
                complexity_term = torch.zeros(1, device=prm.device)

            # Total objective:
            objective = avg_empiric_loss + complexity_term

            # Take gradient step:
            grad_step(objective, optimizer, lr_schedule, prm.lr, i_epoch)

            # Print status:
            log_interval = 1000
            if batch_idx % log_interval == 0:
                batch_acc = correct_rate(outputs, targets)
                print(cmn.status_string(i_epoch, prm.num_epochs, batch_idx, n_batches, batch_acc, objective.item()) +
                      ' Loss: {:.4}\t Comp.: {:.4}'.format(avg_empiric_loss.item(), complexity_term.item()))

        # End batch loop

        # save results for epochs-figure:
        if figure_flag and (i_epoch % prm.log_figure['interval_epochs'] == 0):
             save_result_for_figure(post_model, prior_model, data_loader, prm, log_mat, i_epoch)


    # End run_train_epoch()
    # -------------------------------------------------------------------------------------------
    #  Main Script
    # -------------------------------------------------------------------------------------------


    #  Update Log file
    if verbose:
        write_to_log(cmn.get_model_string(post_model), prm)
        write_to_log('Number of weights: {}'.format(post_model.weights_count), prm)
        write_to_log('Total number of steps: {}'.format(n_batches * prm.num_epochs), prm)
        write_to_log('Number of training samples: {}'.format(data_loader['n_train_samples']), prm)


    start_time = timeit.default_timer()

    if figure_flag:
        n_logs = 1 + ((prm.num_epochs-1) // prm.log_figure['interval_epochs'])
        log_mat = np.zeros((len(prm.log_figure['val_types']), n_logs))

    else:
        log_mat = None

    # Run training epochs:
    for i_epoch in range(prm.num_epochs):
        run_train_epoch(i_epoch, log_mat)

    # evaluate final perfomance on train-set
    train_acc, train_loss = run_eval_Bayes(post_model, train_loader, prm)

    # Test:
    test_acc, test_loss = run_eval_Bayes(post_model, test_loader, prm)
    test_err = 1 - test_acc

    # Log results
    if verbose:
        write_to_log('>Train-err. : {:.4}%\t Train-loss: {:.4}'.format(100*(1-train_acc), train_loss), prm)
        write_to_log('>Test-err. {:1.3}%, Test-loss:  {:.4}'.format(100*(test_err), test_loss), prm)

    stop_time = timeit.default_timer()
    if verbose:
        cmn.write_final_result(test_acc, stop_time - start_time, prm)

    if figure_flag:
        plot_log(log_mat, prm)

    return post_model, test_err, test_loss, log_mat


# -------------------------------------------------------------------------------------------
#  Bound evaluation
# -------------------------------------------------------------------------------------------
def eval_bound(post_model, prior_model, data_loader, prm, avg_empiric_loss=None, dvrg_val=None):


    n_train_samples = data_loader['n_train_samples']

    if not avg_empiric_loss:
        _, avg_empiric_loss = run_eval_Bayes(post_model, data_loader['train'], prm)


    #  complexity/prior term:
    complexity_term = get_task_complexity(
        prm, prior_model, post_model, n_train_samples, avg_empiric_loss, dvrg_val)

    # Total objective:
    bound_val = avg_empiric_loss + complexity_term.item()
    return bound_val


# -------------------------------------------------------------------------------------------

def save_result_for_figure(post_model, prior_model, data_loader, prm, log_mat, i_epoch):
    '''save results for epochs-figure'''
    from Utils.complexity_terms import get_net_densities_divergence
    prm_eval = deepcopy(prm) #   parameters for evaluation
    prm_eval.loss_type = prm.log_figure['loss_type_eval']
    val_types = prm.log_figure['val_types']

    # evaluation
    _, train_loss = run_eval_Bayes(post_model, data_loader['train'], prm_eval)
    _, test_loss = run_eval_Bayes(post_model, data_loader['test'], prm_eval)

    for i_val_type, val_type in enumerate(val_types):
        if val_type[0] == 'i_epoch':
            val = i_epoch
        elif val_type[0] == 'train_loss':
            val = train_loss
        elif val_type[0] == 'test_loss':
            val = test_loss
        elif val_type[0] == 'Bound':
            prm_eval.complexity_type = val_type[1]
            prm_eval.divergence_type = val_type[2]
            val = eval_bound(post_model, prior_model, data_loader, prm_eval, train_loss)
            write_to_log(str(val_type) + ' = ' + str(val), prm)
        elif val_type[0] == 'Divergence':
            prm_eval.divergence_type = val_type[1]
            val = get_net_densities_divergence(prior_model, post_model, prm_eval)
        else:
            raise ValueError('Invalid loss_type_eval')

        log_counter = i_epoch // prm.log_figure['interval_epochs']
        log_mat[i_val_type, log_counter] = val
    # end for val_types

# -------------------------------------------------------------------------------------------

def plot_log(log_mat, prm, val_types_for_show=None, y_axis_lim=None):

    if val_types_for_show is None:
        val_types_for_show = prm.log_figure['val_types']

    x_axis = np.arange(0, prm.num_epochs, prm.log_figure['interval_epochs'])

    # Plot the analysis:
    plt.figure()
    for i_val_type, val_type in enumerate(val_types_for_show):
        if val_type in val_types_for_show:
            plt.plot(x_axis, log_mat[i_val_type],
                         label=str(val_type))

    # plt.xticks(train_samples_vec)
    plt.xlabel('Epoch')
    plt.ylabel(prm.log_figure['loss_type_eval'])
    plt.legend()
    plt.title(prm.result_dir)
    # plt.savefig(root_saved_dir + base_run_name+'.pdf', format='pdf', bbox_inches='tight')
    if y_axis_lim:
        plt.ylim(y_axis_lim)
    plt.grid()
    plt.show()
