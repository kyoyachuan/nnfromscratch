import numpy as np

from .layer import Dense, Layer
from .activation import activation_layer


class Model:
    def __init__(self, loss: Layer = None):
        """
        loss: loss function
        """
        self.layers = []
        self.loss = loss

    def __call__(self, x) -> np.ndarray:
        """
        x: input data
        return: estimated data
        """
        return self.forward(x)

    def add(self, layer: Layer):
        """
        layer: layer
        """
        self.layers.append(layer)

    def forward(self, x) -> np.ndarray:
        """
        x: input data
        return: estimated data
        """
        for layer in self.layers:
            x = layer(x)
        return x

    def backward(self, grad=1):
        """
        grad: gradient
        """
        if self.loss:
            grad = self.loss.backward()
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def parameters(self) -> dict:
        """
        return: parameters dict
        """
        params = {}
        for index, layer in enumerate(self.layers):
            if layer.params:
                for key in layer.params:
                    params[f'{key}_{index}'] = layer.params[key]
        return params

    def __repr__(self) -> str:
        """
        return: model representation
        """
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
        """
        input_size: input data size
        hidden_size_1: hidden layer 1 size
        hidden_size_2: hidden layer 2 size
        output_size: output data size
        activation: activation function
        loss: loss function
        """
        super().__init__(loss)
        self.add(Dense(input_size, hidden_size_1),)
        self.add(activation_layer(activation),)
        self.add(Dense(hidden_size_1, hidden_size_2),)
        self.add(activation_layer(activation),)
        self.add(Dense(hidden_size_2, output_size),)
        self.add(activation_layer('sigmoid'))
