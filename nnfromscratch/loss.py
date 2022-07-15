import numpy as np

from .layer import Layer


class BinaryCrossEntropy(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x, t):
        """
        x: estimated data
        t: ground truth data
        """
        self.input = x
        self.target = t
        self.output = -np.sum(t * np.log(x) + (1 - t) * np.log(1 - x)) / self.input.shape[0]

    def backward(self, grad=1):
        return grad * (-self.target / self.input + (1 - self.target) / (1 - self.input)) / self.input.shape[0]


class MeanSquareError(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x, t):
        """
        x: estimated data
        t: ground truth data
        """
        self.input = x
        self.target = t
        self.output = np.sum((t - x) ** 2) / (2 * self.input.shape[0])

    def backward(self, grad=1):
        return grad * -1 / self.input.shape[0] * (self.target - self.input)
