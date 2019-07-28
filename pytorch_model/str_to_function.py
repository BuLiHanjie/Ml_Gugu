import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error


def str_to_activation(name):
    if name == 'leaky_relu':
        return nn.LeakyReLU(0.1)
    elif name == 'relu':
        return nn.ReLU()
    elif name == 'sigmoid':
        return nn.Sigmoid()
    elif name == 'tanh':
        return nn.Tanh()
    elif name == 'softmax':
        return nn.Softmax()
    return None


def str_top_optimizer(name, params, **kwargs):
    if name == 'adam':
        return optim.Adam(params, **kwargs)
    if name == 'sgd':
        return optim.SGD(params, **kwargs)
    return None


# def str_to_loss(name, **kwargs):
#     if name == ''

def str_to_metric(name):
    if name == 'aus':
        return roc_auc_score
    if name == 'mse':
        return mean_squared_error
    if name == 'mae':
        return mean_absolute_error
    return None