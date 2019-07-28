import numpy as np
import lightgbm as lgb
import xgboost as xgb


def lgb_tencent(y_pred, y_true):
    if isinstance(y_true, lgb.Dataset):
        y_true = y_true.get_label()
    _sum = y_pred + y_true
    grad = np.where(y_true < y_pred, 1. / _sum, -1 / _sum)
    hess = np.where(y_true < y_pred, 1., 1)
    # print(grad.mean())
    return grad, hess


def lgb_tencent_v2(y_pred, y_true):
    if isinstance(y_true, lgb.Dataset):
        y_true = y_true.get_label()
    y_pred[y_pred < 0] = 0.
    _sum = y_pred + y_true
    _sign = np.sign(y_pred - y_true)
    grad = 4 * y_true * _sign / np.square(_sum)
    hess = -8 * y_true * _sign / np.power(_sum, 3)
    hess = np.abs(hess)
    return grad, hess


def xgb_smape(y_pred, y_true):
    if isinstance(y_true, xgb.DMatrix):
        y_true = y_true.get_label()
    y_pred[y_pred < 0] = 0.
    _sum = y_pred + y_true
    _sign = np.sign(y_pred - y_true)
    grad = 4 * y_true * _sign / np.square(_sum)
    hess = -8 * y_true * _sign / np.power(_sum, 3)
    hess = np.abs(hess)
    # hess = -hess
    return grad, hess


def xgb_smape_v2(y_pred, y_true):
    if isinstance(y_true, xgb.DMatrix):
        y_true = y_true.get_label()
    y_pred[y_pred < 0] = 0.
    _minus = y_pred - y_true
    _sum = y_pred + y_true
    _sinh_minus = np.sinh(_minus)
    _cosh_minus = np.cosh(_minus)
    grad = 2 * _sinh_minus / (_sum * _cosh_minus) - 2 * np.log(_cosh_minus) / (np.square(_sum))
    hess = \
        4 * np.log(_cosh_minus) / np.power(_sum, 3) - \
        2 * np.square(_sinh_minus) / (_sum * np.square(_cosh_minus)) - \
        4 * _sinh_minus / (np.square(_sum) * _cosh_minus) + \
        2 / _sum
    hess[hess < 0] = 0
    return grad, hess
