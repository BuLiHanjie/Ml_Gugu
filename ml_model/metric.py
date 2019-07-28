import numpy as np
import lightgbm as lgb
import xgboost as xgb


def tencent_mape(y_pred, y_true):
    if isinstance(y_true, lgb.Dataset):
        y_true = y_true.get_label()
    y_pred[y_pred < 1] = 1.
    s1 = np.abs(y_true - y_pred)
    s2 = y_true + y_pred
    s2[s2 == 0] = 1e-3
    res = np.mean(s1 / s2)
    return 'tencent_mape', res, False

def xgb_smape_metric(y_pred, y_true):
    if isinstance(y_true, xgb.DMatrix):
        y_true = y_true.get_label()
    y_pred[y_pred < 1] = 1.
    s1 = np.abs(y_true - y_pred)
    s2 = y_true + y_pred
    s2[s2 == 0] = 1e-3
    res = np.mean(s1 / s2)
    return 'smape', res
