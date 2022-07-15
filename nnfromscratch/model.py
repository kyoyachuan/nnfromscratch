import numpy as np

from .layer import Dense, Layer, Parameter
from .activation import activation_layer


class Model:
    def __init__(self, loss=None):
        self.layers = []
        self.loss = loss

    def __call__(self, x):
        return self.forward(x)

    def add(self, layer: Layer):
        self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def backward(self, grad=1):
        if self.loss:
            grad = self.loss.backward()
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def parameters(self):
        params = {}
        for index, layer in enumerate(self.layers):
            if layer.params:
                for key in layer.params:
                    params[f'{key}_{index}'] = layer.params[key]
        return params

    def __repr__(self):
        return 'Model({})'.format(self.layers)


class TwoLayerNetwork(Model):
    def __init__(
        self,
        input_size: int,
        hidden_size_1: int,
        hidden_size_2: int,
        output_size: int,
        activation: str = 'sigmoid',
        loss=None
    ):
        super().__init__(loss)
        self.add(Dense(input_size, hidden_size_1),)
        self.add(activation_layer(activation),)
        self.add(Dense(hidden_size_1, hidden_size_2),)
        self.add(activation_layer(activation),)
        self.add(Dense(hidden_size_2, output_size),)
        self.add(activation_layer('sigmoid'))