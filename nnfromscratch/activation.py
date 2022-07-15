import numpy as np

from .layer import Layer


def activation_layer(activation):
    if activation == 'sigmoid':
        return Sigmoid()
    elif activation == 'relu':
        return Relu()
    elif activation == 'linear':
        return Linear()
    elif activation == 'tanh':
        return Tanh()
    else:
        raise ValueError('Unknown activation: {}'.format(activation))


class Linear(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        self.input = x
        self.output = x

    def backward(self, grad):
        return grad * 1


class Sigmoid(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        self.input = x
        self.output = 1 / (1 + np.exp(-x))

    def backward(self, grad):
        return grad * (1 - self.output) * self.output


class Relu(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        self.input = x
        self.output = np.maximum(0, x)

    def backward(self, grad):
        return grad * (self.input > 0)


class Tanh(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        self.input = x
        self.output = np.tanh(x)

    def backward(self, grad):
        return grad * (1 - self.output ** 2)

