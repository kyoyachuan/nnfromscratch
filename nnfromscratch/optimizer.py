import numpy as np


class Optimizer:
    def __init__(self, params: dict, lr=0.01):
        self.params = params
        self.lr = lr

    def step(self):
        raise NotImplementedError

    def zero_grad(self):
        for key in self.params:
            self.params[key].grad = np.zeros_like(self.params[key].data)


class SGD(Optimizer):
    def __init__(self, params: dict, lr=0.01):
        super().__init__(params, lr)

    def step(self):
        for key in self.params:
            self.params[key].data -= self.lr * self.params[key].grad


class Momentum(Optimizer):
    def __init__(self, params: dict, lr=0.01, momentum=0.9):
        super().__init__(params, lr)
        self.momentum = momentum
        self.v = {}
        for key in self.params:
            self.v[key] = np.zeros_like(self.params[key].data)

    def step(self):
        for key in self.params:
            self.v[key] = self.momentum * self.v[key] - self.lr * self.params[key].grad
            self.params[key].data += self.v[key]