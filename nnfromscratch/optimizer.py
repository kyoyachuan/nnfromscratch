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


class Adam(Optimizer):
    def __init__(self, params: dict, lr=0.01, beta1=0.9, beta2=0.999):
        super().__init__(params, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.m = {}
        self.v = {}
        self.iter = 1
        for key in self.params:
            self.m[key] = np.zeros_like(self.params[key].data)
            self.v[key] = np.zeros_like(self.params[key].data)

    def step(self):
        for key in self.params:
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * self.params[key].grad
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (self.params[key].grad ** 2)
            m_hat = self.m[key] / (1 - self.beta1 ** self.iter)
            v_hat = self.v[key] / (1 - self.beta2 ** self.iter)
            self.params[key].data -= self.lr * m_hat / (np.sqrt(v_hat) + 1e-7)

    def zero_grad(self):
        super().zero_grad()
        self.iter += 1

    def __del__(self):
        self.iter = 1