
from __future__ import absolute_import, division, print_function

import timeit

from Models.stochastic_models import get_model
from Utils import common as cmn, data_gen
from Utils.Bayes_utils import run_eval_Bayes
from Utils.complexity_terms import get_task_complexity
from Utils.common import grad_step, count_correct, write_to_log
from Utils.Losses import get_loss_func


def run_learning(task_data, prior_model, prm, init_from_prior=True, verbose=1):

    # -------------------------------------------------------------------------------------------
    #  Setting-up
    # -------------------------------------------------------------------------------------------
    # Unpack parameters:
    optim_func, optim_args, lr_schedule =\
        prm.optim_func, prm.optim_args, prm.lr_schedule

    # Loss criterion
    loss_criterion = get_loss_func(prm)

    # Create posterior model for the new task:
    post_model = get_model(prm)

    if init_from_prior:
        post_model.load_state_dict(prior_model.state_dict())

        # prior_model_dict = prior_model.state_dict()
        # post_model_dict = post_model.state_dict()
        #
        # # filter out unnecessary keys:
        # prior_model_dict = {k: v for k, v in prior_model_dict.items() if '_log_var' in k or '_mu' in k}
        # # overwrite entries in the existing state dict:
        # post_model_dict.update(prior_model_dict)
        #
        # # #  load the new state dict
        # post_model.load_state_dict(post_model_dict)

        # add_noise_to_model(post_model, prm.kappa_factor)

    # The data-sets of the new task:
    train_loader = task_data['train']
    test_loader = task_data['test']
    n_train_samples = len(train_loader.dataset)
    n_batches = len(train_loader)

    #  Get optimizer:
    optimizer = optim_func(post_model.parameters(), **optim_args)


    # -------------------------------------------------------------------------------------------
    #  Training epoch  function
    # -------------------------------------------------------------------------------------------

    def run_train_epoch(i_epoch):
        log_interval = 500

        post_model.train()

        for batch_idx, batch_data in enumerate(train_loader):

            # get batch data:
            inputs, targets = data_gen.get_batch_vars(batch_data, prm)
            batch_size = inputs.shape[0]

            correct_count = 0
            sample_count = 0

            # Monte-Carlo iterations:
            n_MC = prm.n_MC
            avg_empiric_loss = 0
            complexity_term = 0

            for i_MC in range(n_MC):

                # Calculate empirical loss:
                outputs = post_model(inputs)
                avg_empiric_loss_curr = (1 / batch_size) * loss_criterion(outputs, targets)

                # complexity_curr = get_task_complexity(prm, prior_model, post_model,
                #                                            n_train_samples, avg_empiric_loss_curr)

                avg_empiric_loss += (1 / n_MC) * avg_empiric_loss_curr
                # complexity_term += (1 / n_MC) * complexity_curr

                correct_count += count_correct(outputs, targets)
                sample_count += inputs.size(0)
            # end monte-carlo loop

            complexity_term = get_task_complexity(prm, prior_model, post_model,  n_train_samples, avg_empiric_loss)

            # Approximated total objective (for current batch):
            if prm.complexity_type == 'Variational_Bayes':
                # note that avg_empiric_loss_per_task is estimated by an average over batch samples,
                #  but its weight in the objective should be considered by how many samples there are total in the task
                total_objective = avg_empiric_loss * (n_train_samples) + complexity_term
            else:
                total_objective = avg_empiric_loss + complexity_term

            # Take gradient step with the posterior:
            grad_step(total_objective, optimizer, lr_schedule, prm.lr, i_epoch)


            # Print status:
            if batch_idx % log_interval == 0:
                batch_acc = correct_count / sample_count
                print(cmn.status_string(i_epoch, prm.n_meta_test_epochs, batch_idx, n_batches, batch_acc, total_objective.item()) +
                      ' Empiric Loss: {:.4}\t Intra-Comp. {:.4}'.
                      format(avg_empiric_loss.item(), complexity_term.item()))
        # end batch loop
    # end run_train_epoch()


    # -----------------------------------------------------------------------------------------------------------#
    # Update Log file
    if verbose == 1:
        write_to_log('Total number of steps: {}'.format(n_batches * prm.n_meta_test_epochs), prm)

    # -------------------------------------------------------------------------------------------
    #  Run epochs
    # -------------------------------------------------------------------------------------------
    start_time = timeit.default_timer()

    # Training loop:
    for i_epoch in range(prm.n_meta_test_epochs):
        run_train_epoch(i_epoch)

    # Test:
    test_acc, test_loss = run_eval_Bayes(post_model, test_loader, prm)

    stop_time = timeit.default_timer()
    cmn.write_final_result(test_acc, stop_time - start_time, prm, result_name=prm.test_type, verbose=verbose)

    test_err = 1 - test_acc
    return test_err, post_model
