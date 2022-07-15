import numpy as np


class Parameter:
    def __init__(self, data: np.ndarray, grad: np.ndarray):
        self.data = data
        self.grad = grad


class Layer:
    def __init__(self):
        self.params = {}

    def __call__(self, x, **kwargs):
        self.forward(x, **kwargs)
        return self.output

    def forward(self, x):
        raise NotImplementedError

    def backward(self, grad):
        raise NotImplementedError


class Dense(Layer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.params['W'] = Parameter(
            np.random.randn(input_size, output_size),
	    np.zeros((input_size, output_size))
	)
        self.params['b'] = Parameter(
            np.random.randn(1, output_size),
	    np.zeros((1, output_size))
	)

    def forward(self, x):
        self.input = x
        self.output = np.dot(x, self.params['W'].data) + self.params['b'].data

    def backward(self, grad):
        self.params['W'].grad = np.dot(self.input.T, grad)
        self.params['b'].grad = np.sum(grad, axis=0)
        return np.dot(grad, self.params['W'].data.T)
