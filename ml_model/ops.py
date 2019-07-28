import numpy as np


def data_copy(x, y, k):
    negative_x, postive_x = x[y == 0], x[y == 1]
    negative_y, postive_y = y[y == 0], y[y == 1]
    return np.concatenate([negative_x] + [postive_x] * k, axis=0), np.concatenate([negative_y] + [postive_y] * k, axis=0)