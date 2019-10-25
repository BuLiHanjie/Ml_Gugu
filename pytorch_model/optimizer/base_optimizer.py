class BaseOptimizer:

    def __init__(self):
        self.optimizer = None

    def step(self):
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def add_param(self, param):
        self.optimizer.add_param_group(
            {
                'params': param
            }
        )

    def state_dict(self):
        return self.optimizer.state_dict()