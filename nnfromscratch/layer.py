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


class Convolution2D(Layer):
    def __init__(self, input_size, output_size, kernel_size):
        super().__init__()
        self.params['W'] = Parameter(
            np.random.randn(output_size, input_size, kernel_size, kernel_size),
            np.zeros((output_size, input_size, kernel_size, kernel_size))
        )
        self.params['b'] = Parameter(
            np.random.randn(1, output_size),
            np.zeros((1, output_size))
        )
        self.kernel_size = kernel_size

    def forward(self, x):
        self.input = x
        self.output = np.zeros((x.shape[0], x.shape[1], x.shape[2], x.shape[3]))
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                for k in range(x.shape[2]):
                    for l in range(x.shape[3]):
                        self.output[i, j, k, l] = np.sum(
                            self.params['W'].data[:, i, k:k + self.kernel_size, l:l + self.kernel_size] * x[i, j, k:k + self.kernel_size, l:l + self.kernel_size]
                        ) + self.params['b'].data[0, j]

    def backward(self, grad):
        self.params['W'].grad = np.zeros(self.params['W'].data.shape)
        self.params['b'].grad = np.zeros(self.params['b'].data.shape)
        for i in range(grad.shape[0]):
            for j in range(grad.shape[1]):
                for k in range(grad.shape[2]):
                    for l in range(grad.shape[3]):
                        self.params['W'].grad[:, i, k:k + self.kernel_size, l:l + self.kernel_size] += grad[i, j, k, l] * self.input[i, j, k:k + self.kernel_size, l:l + self.kernel_size]
                        self.params['b'].grad[0, j] += grad[i, j, k, l]
        return np.sum(grad * self.params['W'].data, axis=(1, 2, 3))