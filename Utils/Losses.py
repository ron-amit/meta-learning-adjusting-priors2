from __future__ import absolute_import, division, print_function

import torch
from torch.nn.modules.module import Module
import torch.nn as nn
import math
from Utils.data_gen import get_info
# -----------------------------------------------------------------------------------------------------------#
# Returns loss function
# -----------------------------------------------------------------------------------------------------------#
def get_loss_func(prm):
    # Note: 1. the loss function use the un-normalized net outputs (scores, not probabilities)
    #       2. The returned loss is summed (not averaged) over samples!!!

    loss_type = prm.loss_type
    task_info = get_info(prm)
    task_type = task_info['type']

    if loss_type == 'CrossEntropy':
        assert task_type == 'multi_class'
        return nn.CrossEntropyLoss(reduction='sum')

    elif loss_type == 'L2_SVM':
        assert task_type == 'multi_class'
        return nn.MultiMarginLoss(p=2, margin=1, weight=None, reduction='sum')

    elif loss_type == 'Logistic_binary':
        assert task_type == 'binary_class'
        return Logistic_Binary_Loss(reduction='sum')

    elif loss_type == 'Logistic_Binary_Clipped':
        assert task_type == 'binary_class'
        return Logistic_Binary_Loss_Clipped(reduction='sum')


    elif loss_type == 'Zero_One':
        if task_type == 'binary_class':
            return Zero_One_Binary(reduction='sum')
        elif  task_type == 'multi_class':
            return Zero_One_Multi(reduction='sum')

    raise ValueError('Invalid loss_type')



# -----------------------------------------------------------------------------------------------------------#
# Definitions of loss functions
# -----------------------------------------------------------------------------------------------------------#

def _assert_no_grad(variable):
    assert not variable.requires_grad, \
        "nn criterions don't compute the gradient w.r.t. targets - please " \
        "mark these variables as volatile or not requiring gradients"
# -----------------------------------------------------------------------------------------------------------#
# Base class for loss functions
class _Loss(Module):
    def __init__(self, reduction='sum'):
        super(_Loss, self).__init__()
        self.reduction = reduction

# -----------------------------------------------------------------------------------------------------------#

class Logistic_Binary_Loss(_Loss):
    r"""Creates a criterion that optimizes a two-class classification
    logistic loss between input `x` (a 2D mini-batch Tensor) and
    target `y` (which is a tensor containing either `1` or `-1`).

    ::

        loss(x, y) = sum_i (log(1 + exp(-y[i]*x[i]))) / x.nelement()  / math.log(2)

    The normalization by the number of elements in the input can be disabled by
    setting `self.size_average` to ``False``.
    """

    def forward(self, input, target):
        # validity checks
        _assert_no_grad(target)
        # assert input.shape[1] == 1 # this loss works only for binary classification
        input = input[:, 0]
        assert self.reduction == 'sum'
        # switch labels to {-1,1}
        target = target.float() * 2 - 1
        loss_vec = torch.log(1 + torch.exp(-target * input)) / math.log(2)
        loss_sum = loss_vec.sum()
        return loss_sum
# -----------------------------------------------------------------------------------------------------------#

class Logistic_Binary_Loss_Clipped(_Loss):
    r"""Creates a criterion that optimizes a two-class classification
    logistic loss between input `x` (a 2D mini-batch Tensor) and
    target `y` (which is a tensor containing either `1` or `-1`).

    ::

        loss(x, y) = sum_i (log(1 + exp(-y[i]*x[i]))) / x.nelement()  / math.log(2)

    The normalization by the number of elements in the input can be disabled by
    setting `self.size_average` to ``False``.
    """

    def forward(self, input, target):
        # validity checks
        _assert_no_grad(target)
        # assert input.shape[1] == 1 # this loss works only for binary classification
        input = input[:, 0]
        assert self.reduction == 'sum'
        # switch labels to {-1,1}
        target = target.float() * 2 - 1
        loss_vec = torch.log(1 + torch.exp(-target * input)) / math.log(2)
        # clamp\clipp values to [0,1]:
        loss_vec = loss_vec.clamp(0, 1)
        loss_sum = loss_vec.sum()
        return loss_sum

# -----------------------------------------------------------------------------------------------------------#

class Zero_One_Binary(_Loss):
    # zero one-loss of binary classifier with labels {-1,1}
    def forward(self, input, target):
        # validity checks
        _assert_no_grad(target)
        # assert input.shape[1] == 1 # this loss works only for binary classification
        input = input[:, 0]
        assert self.reduction == 'sum'
        # switch labels to {-1,1}
        target = target.float() * 2 - 1
        loss_sum = (target != torch.sign(input)).sum().float()
        return loss_sum

# -----------------------------------------------------------------------------------------------------------#

class Zero_One_Multi(_Loss):
    # zero one-loss of multi-class classifier
    # labels are {0,1,..,n_classes-1}
    # inputs are the class scores outputed by the network
    def forward(self, input, target):
        # validity checks
        _assert_no_grad(target)
        # assert input.shape[1] > 1  # this loss works only for multi-class classification
        label_est = input.argmax(dim=1)
        assert self.reduction == 'sum'
        loss_sum = (target != label_est).sum().float()
        return loss_sum

# -----------------------------------------------------------------------------------------------------------#
