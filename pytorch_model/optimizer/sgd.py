from torch import optim
from .base_optimizer import BaseOptimizer


class SGD(BaseOptimizer):

    def __init__(self, params, lr=0.01, momentum=0.9):
        super().__init__()
        self.optimizer = optim.SGD(params, lr, momentum)
