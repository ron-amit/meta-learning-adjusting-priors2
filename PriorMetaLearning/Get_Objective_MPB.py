from __future__ import absolute_import, division, print_function

import torch
from Utils import data_gen
from Utils.complexity_terms import get_task_complexity, get_meta_complexity_term, get_hyper_divergnce
from Utils.common import count_correct

# -------------------------------------------------------------------------------------------
#
# -------------------------------------------------------------------------------------------
def get_objective(prior_model, prm, mb_data_loaders, mb_iterators, mb_posteriors_models, loss_criterion, n_train_tasks):
    '''  Calculate objective based on tasks in meta-batch '''
    # note: it is OK if some tasks appear several times in the meta-batch

    n_tasks_in_mb = len(mb_data_loaders)

    correct_count = 0
    sample_count = 0

    # Hyper-prior term:
    hyper_dvrg = get_hyper_divergnce(prm, prior_model)
    meta_complex_term = get_meta_complexity_term(hyper_dvrg, prm, n_train_tasks)


    avg_empiric_loss_per_task = torch.zeros(n_tasks_in_mb, device=prm.device)
    complexity_per_task = torch.zeros(n_tasks_in_mb, device=prm.device)
    n_samples_per_task = torch.zeros(n_tasks_in_mb, device=prm.device)  # how many sampels there are total in each task (not just in a batch)

    # ----------- loop over tasks in meta-batch -----------------------------------#
    for i_task in range(n_tasks_in_mb):

        n_samples = mb_data_loaders[i_task]['n_train_samples']
        n_samples_per_task[i_task] = n_samples

        # get sample-batch data from current task to calculate the empirical loss estimate:
        batch_data = data_gen.get_next_batch_cyclic(mb_iterators[i_task], mb_data_loaders[i_task]['train'])

        # get batch variables:
        inputs, targets = data_gen.get_batch_vars(batch_data, prm)
        batch_size = inputs.shape[0]

        # The posterior model corresponding to the task in the batch:
        post_model = mb_posteriors_models[i_task]
        post_model.train()

        # Monte-Carlo iterations:
        n_MC = prm.n_MC

        avg_empiric_loss = 0.0
        complexity = 0.0

        # Monte-Carlo loop
        for i_MC in range(n_MC):

            # Debug
            # print(targets[0].data[0])  # print first image label
            # import matplotlib.pyplot as plt
            # plt.imshow(inputs[0].cpu().data[0].numpy())  # show first image
            # plt.show()

            # Empirical Loss on current task:
            outputs = post_model(inputs)
            avg_empiric_loss_curr = (1 / batch_size) * loss_criterion(outputs, targets)

            correct_count += count_correct(outputs, targets)  # for print
            sample_count += inputs.size(0)

            # Intra-task complexity of current task:
            # curr_complexity = get_task_complexity(prm, prior_model, post_model,
            #     n_samples, avg_empiric_loss_curr, hyper_dvrg, n_train_tasks=n_train_tasks, noised_prior=True)

            avg_empiric_loss += (1 / n_MC) * avg_empiric_loss_curr
            # complexity +=  (1 / n_MC) * curr_complexity
        # end Monte-Carlo loop

        complexity = get_task_complexity(prm, prior_model, post_model,
                                         n_samples, avg_empiric_loss, hyper_dvrg,
                                         n_train_tasks=n_train_tasks, noised_prior=True)
        avg_empiric_loss_per_task[i_task] = avg_empiric_loss
        complexity_per_task[i_task] = complexity
    # end loop over tasks in meta-batch


    # Approximated total objective:
    if prm.complexity_type == 'Variational_Bayes':
        # note that avg_empiric_loss_per_task is estimated by an average over batch samples,
        #  but its weight in the objective should be considered by how many samples there are total in the task
        total_objective =\
            (avg_empiric_loss_per_task * n_samples_per_task + complexity_per_task).mean() * n_train_tasks + meta_complex_term
        # total_objective = ( avg_empiric_loss_per_task * n_samples_per_task + complexity_per_task).mean() + meta_complex_term

    else:
        total_objective =\
            avg_empiric_loss_per_task.mean() + complexity_per_task.mean() + meta_complex_term

    info = {'sample_count': sample_count, 'correct_count': correct_count,
                  'avg_empirical_loss': avg_empiric_loss_per_task.mean().item(),
                  'avg_intra_task_comp': complexity_per_task.mean().item(),
                  'meta_comp': meta_complex_term.item()}
    return total_objective, info
