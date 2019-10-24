from torch import optim
from .base_optimizer import BaseOptimizer


class Adagrad(BaseOptimizer):

    def __init__(self, params, lr=0.01, lr_decay=0, weight_decay=0, initial_accumulator_value=0):
        super().__init__()
        self.optimizer = optim.Adagrad(
            params,
            lr=lr,
            lr_decay=lr_decay,
            weight_decay=weight_decay,
            initial_accumulator_value=initial_accumulator_value
        )
