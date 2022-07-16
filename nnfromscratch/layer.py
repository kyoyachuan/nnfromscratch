import numpy as np


class Parameter:
    def __init__(self, data: np.ndarray, grad: np.ndarray):
        """
        data: parameter data
        grad: parameter grad
        """
        self.data = data
        self.grad = grad


class Layer:
    def __init__(self):
        self.params = {}

    def __call__(self, x, **kwargs) -> np.ndarray:
        """
        x: input data
        return: estimated data
        """
        self.forward(x, **kwargs)
        return self.output

    def forward(self, x):
        raise NotImplementedError

    def backward(self, grad):
        raise NotImplementedError


class Dense(Layer):
    def __init__(self, input_size, output_size):
        """
        input_size: input data size
        output_size: output data size
        """
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
        """
        x: input data
        """
        self.input = x
        self.output = np.dot(x, self.params['W'].data) + self.params['b'].data

    def backward(self, grad) -> np.ndarray:
        """
        grad: gradient
        return: gradient w.r.t to input data
        """
        self.params['W'].grad = np.dot(self.input.T, grad)
        self.params['b'].grad = np.sum(grad, axis=0)
        return np.dot(grad, self.params['W'].data.T)


class Convolution2D(Layer):
    def __init__(self, input_size, output_size, kernel_size):
        """
        input_size: input data channel size
        output_size: output data channel size
        kernel_size: kernel size
        """
        super().__init__()
        self.params['W'] = Parameter(
            np.random.randn(input_size, output_size, kernel_size, kernel_size),
            np.zeros((input_size, output_size, kernel_size, kernel_size))
        )
        self.params['b'] = Parameter(
            np.random.randn(1, output_size),
            np.zeros((1, output_size))
        )
        self.kernel_size = kernel_size

    def _get_output_shape(self, input_size) -> int:
        """
        input_size: input data size
        return: output data size
        """
        return (input_size - self.kernel_size) // 1 + 1

    def forward(self, x):
        """
        x: input data
        """
        self.input = x
        self.output = np.zeros(
            (x.shape[0],
             self.params['W'].data.shape[1],
             self._get_output_shape(x.shape[2]),
             self._get_output_shape(x.shape[3]))
        )
        for i in range(x.shape[0]):
            for j in range(self.params['W'].data.shape[1]):
                for k in range(self._get_output_shape(x.shape[2])):
                    for l in range(self._get_output_shape(x.shape[3])):
                        self.output[i, j, k, l] = np.sum(
                            self.params['W'].data[:, j, k: k + self.kernel_size, l: l + self.kernel_size] *
                            x[i, :, k: k + self.kernel_size, l: l + self.kernel_size]
                        ) + self.params['b'].data[0, j]

    def backward(self, grad) -> np.ndarray:
        """
        grad: gradient
        return: gradient w.r.t to input data
        """
        self.params['W'].grad = np.zeros(self.params['W'].data.shape)
        self.params['b'].grad = np.zeros(self.params['b'].data.shape)
        for i in range(grad.shape[0]):
            for j in range(grad.shape[1]):
                for k in range(grad.shape[2]):
                    for l in range(grad.shape[3]):
                        self.params['W'].grad[:, j, k:k + self.kernel_size, l:l + self.kernel_size] += \
                            grad[i, j, k, l] * self.input[i, :, k:k + self.kernel_size, l:l + self.kernel_size]
                        self.params['b'].grad[0, j] += grad[i, j, k, l]
        return np.sum(grad * self.params['W'].data, axis=(1, 2, 3))
