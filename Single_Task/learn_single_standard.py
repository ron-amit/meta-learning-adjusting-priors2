
from __future__ import absolute_import, division, print_function

import timeit

from Models.deterministic_models import get_model
from Utils import common as cmn, data_gen
from Utils.common import count_correct, grad_step, correct_rate, write_to_log
from Utils.Losses import get_loss_func


def run_learning(data_loader, prm, verbose=1, initial_model=None):

    # Unpack parameters:
    optim_func, optim_args, lr_schedule = \
        prm.optim_func, prm.optim_args, prm.lr_schedule

    # Loss criterion
    loss_criterion = get_loss_func(prm)

    # The data-sets:
    train_loader = data_loader['train']
    test_loader = data_loader['test']

    n_batches = len(train_loader)

    # Create  model:
    if hasattr(prm, 'func_model') and prm.func_model:
        import Models.deterministic_models as func_models
        model = func_models.get_model(prm)
    else:
        model = get_model(prm)

    # Load initial weights:
    if initial_model:
        model.load_state_dict(initial_model.state_dict())

    # Gather modules list:
    modules_list = list(model.named_children())
    if hasattr(model, 'net'):
        # extract the modules from 'net' field:
        modules_list += list(model.net.named_children())
        modules_list = [m for m in modules_list if m[0] is not 'net']

    # Determine which parameters are optimized and which are frozen:
    if hasattr(prm, 'freeze_list'):
        freeze_list = prm.freeze_list
        optimized_modules = [named_module[1]
                             for named_module in modules_list
                             if not named_module[0] in freeze_list]
        optimized_params = sum([list(mo.parameters()) for mo in optimized_modules], [])
    elif hasattr(prm, 'not_freeze_list'):
        not_freeze_list = prm.not_freeze_list
        optimized_modules = [named_module[1]
                             for named_module in modules_list
                             if named_module[0] in not_freeze_list]
        optimized_params = sum([list(mo.parameters()) for mo in optimized_modules], [])
    else:
        optimized_params = model.parameters()

    #  Get optimizer:
    optimizer = optim_func(optimized_params, **optim_args)

    # -------------------------------------------------------------------------------------------
    #  Training epoch  function
    # -------------------------------------------------------------------------------------------

    def run_train_epoch(i_epoch):
        log_interval = 500

        model.train()
        for batch_idx, batch_data in enumerate(train_loader):

            # get batch:
            inputs, targets = data_gen.get_batch_vars(batch_data, prm)
            batch_size = inputs.shape[0]

            # Calculate loss:
            outputs = model(inputs)
            loss = (1 / batch_size) * loss_criterion(outputs, targets)

            # Take gradient step:
            grad_step(loss, optimizer, lr_schedule, prm.lr, i_epoch)

            # Print status:
            if batch_idx % log_interval == 0:
                batch_acc = correct_rate(outputs, targets)
                print(cmn.status_string(i_epoch, prm.num_epochs, batch_idx, n_batches, batch_acc, loss.item()))

    # -----------------------------------------------------------------------------------------------------------#
    # Update Log file
    # -----------------------------------------------------------------------------------------------------------#
    update_file = not verbose == 0
    write_to_log(cmn.get_model_string(model), prm, update_file=update_file)
    write_to_log('Number of weights: {}'.format(model.weights_count), prm, update_file=update_file)
    write_to_log('Total number of steps: {}'.format(n_batches * prm.num_epochs), prm, update_file=update_file)
    write_to_log('Number of training samples: {}'.format(data_loader['n_train_samples']), prm, update_file=update_file)

    # -------------------------------------------------------------------------------------------
    #  Run epochs
    # -------------------------------------------------------------------------------------------
    start_time = timeit.default_timer()

    # Training loop:
    for i_epoch in range(prm.num_epochs):
        run_train_epoch(i_epoch)

    # Test:
    test_acc = run_test(model, test_loader, loss_criterion, prm)

    stop_time = timeit.default_timer()
    cmn.write_final_result(test_acc, stop_time - start_time, prm, verbose=verbose, result_name='Standard')

    test_err = 1 - test_acc
    return test_err, model


# -------------------------------------------------------------------------------------------
#  Test evaluation function
# --------------------------------------------------------------------------------------------
def run_test(model, test_loader, loss_criterion, prm):
    model.eval()
    test_loss = 0
    n_correct = 0
    for batch_data in test_loader:
        inputs, targets = data_gen.get_batch_vars(batch_data, prm)
        batch_size = inputs.shape[0]
        outputs = model(inputs)
        test_loss += (1 / batch_size) * loss_criterion(outputs, targets).item()  # sum the mean loss in batch
        n_correct += count_correct(outputs, targets)

    n_test_samples = len(test_loader.dataset)
    n_test_batches = len(test_loader)
    test_loss = test_loss / n_test_batches
    test_acc = n_correct / n_test_samples
    print('\n Standard learning: test loss: {:.4}, test err: {:.3} ( {}/{})\n'.format(
        test_loss, 1-test_acc, n_correct, n_test_samples))
    return test_acc
