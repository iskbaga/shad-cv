import numpy as np

from interface import *


# ================================= 1.4.1 SGD ================================
class SGD(Optimizer):
    def __init__(self, lr):
        self.lr = lr

    def get_parameter_updater(self, parameter_shape):
        """
        :param parameter_shape: tuple, the shape of the associated parameter

        :return: the updater function for that parameter
        """

        def updater(parameter, parameter_grad):
            """
            :param parameter: np.array, current parameter values
            :param parameter_grad: np.array, current gradient, dLoss/dParam

            :return: np.array, new parameter values
            """
            return parameter - self.lr * parameter_grad

        return updater


# ============================= 1.4.2 SGDMomentum ============================
class SGDMomentum(Optimizer):
    def __init__(self, lr, momentum=0.0):
        self.lr = lr
        self.momentum = momentum

    def get_parameter_updater(self, parameter_shape):
        """
        :param parameter_shape: tuple, the shape of the associated parameter

        :return: the updater function for that parameter
        """

        def updater(parameter, parameter_grad):
            """
            :param parameter: np.array, current parameter values
            :param parameter_grad: np.array, current gradient, dLoss/dParam

            :return: np.array, new parameter values
            """
            updater.inertia = self.momentum * updater.inertia + self.lr * parameter_grad
            return parameter - updater.inertia

        updater.inertia = np.zeros(parameter_shape)
        return updater


# ================================ 2.1.1 ReLU ================================
class ReLU(Layer):
    def forward_impl(self, inputs):
        """
        :param inputs: np.array((n, ...)), input values

        :return: np.array((n, ...)), output values

            n - batch size
            ... - arbitrary shape (the same for input and output)
        """
        outputs = np.zeros_like(inputs)
        outputs[inputs > 0] = inputs[inputs > 0]
        return outputs

    def backward_impl(self, grad_outputs):
        """
        :param grad_outputs: np.array((n, ...)), dLoss/dOutputs

        :return: np.array((n, ...)), dLoss/dInputs

            n - batch size
            ... - arbitrary shape (the same for input and output)
        """
        backprop_grads = np.zeros_like(grad_outputs)
        backprop_grads[self.forward_inputs >= 0] = grad_outputs[self.forward_inputs >= 0]
        return backprop_grads


# =============================== 2.1.2 Softmax ==============================
class Softmax(Layer):
    def forward_impl(self, inputs):
        """
        :param inputs: np.array((n, d)), input values

        :return: np.array((n, d)), output values

            n - batch size
            d - number of units
        """
        exp_inputs = np.exp(inputs - np.max(inputs, axis=1)[:, np.newaxis])
        outputs = exp_inputs / np.sum(exp_inputs, axis=1)[:, np.newaxis]
        return outputs

    def backward_impl(self, grad_outputs):
        """
        :param grad_outputs: np.array((n, d)), gradient of the loss with respect to the outputs
        :return: np.array((n, d)), gradient of the loss with respect to the inputs
        """
        n = self.forward_outputs.shape[0]
        backprop_grads = np.zeros_like(self.forward_outputs)
        for i in range(n):
            diag = np.diag(self.forward_outputs[i])
            backprop_grads[i] = (grad_outputs[i] @ (diag - np.outer(self.forward_outputs[i], self.forward_outputs[i])))
        return backprop_grads


# ================================ 2.1.3 Dense ===============================
class Dense(Layer):
    def __init__(self, units, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_units = units

        self.weights, self.weights_grad = None, None
        self.biases, self.biases_grad = None, None

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        (input_units,) = self.input_shape
        output_units = self.output_units

        # Register weights and biases as trainable parameters
        # Note, that the parameters and gradients *must* be stored in
        # self.<p> and self.<p>_grad, where <p> is the name specified in
        # self.add_parameter

        self.weights, self.weights_grad = self.add_parameter(
            name="weights",
            shape=(output_units, input_units),
            initializer=he_initializer(input_units),
        )

        self.biases, self.biases_grad = self.add_parameter(
            name="biases",
            shape=(output_units,),
            initializer=np.zeros,
        )

        self.output_shape = (output_units,)

    def forward_impl(self, inputs):
        """
        :param inputs: np.array((n, d)), input values

        :return: np.array((n, c)), output values

            n - batch size
            d - number of input units
            c - number of output units
        """
        self.forward_inputs = inputs
        self.forward_outputs = inputs @ self.weights.T + self.biases
        return self.forward_outputs

    def backward_impl(self, grad_outputs):
        """
        :param grad_outputs: np.array((n, c)), dLoss/dOutputs

        :return: np.array((n, d)), dLoss/dInputs

            n - batch size
            d - number of input units
            c - number of output units
        """
        self.weights_grad = grad_outputs.T @ self.forward_inputs
        self.biases_grad = np.sum(grad_outputs, axis=0)

        return grad_outputs @ self.weights


# ============================ 2.2.1 Crossentropy ============================
class CategoricalCrossentropy(Loss):
    def value_impl(self, y_gt, y_pred):
        """
        :param y_gt: np.array((n, d)), ground truth (correct) labels
        :param y_pred: np.array((n, d)), estimated target values

        :return: np.array((1,)), mean Loss scalar for batch

            n - batch size
            d - number of units
        """
        y_pred = np.clip(y_pred, eps, 1 - eps)
        loss = -np.sum(y_gt * np.log(y_pred), axis=1)

        return np.mean(loss, keepdims=True)

    def gradient_impl(self, y_gt, y_pred):
        """
        :param y_gt: np.array((n, d)), ground truth (correct) labels
        :param y_pred: np.array((n, d)), estimated target values

        :return: np.array((n, d)), dLoss/dY_pred

            n - batch size
            d - number of units
        """
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return (-1 / y_gt.shape[0]) * (y_gt / y_pred)


# ======================== 2.3 Train and Test on MNIST =======================
def train_mnist_model(x_train, y_train, x_valid, y_valid):
    # your code here \/
    # 1) Create a Model
    optimizer = SGDMomentum(lr=0.01, momentum=0.9)
    loss = CategoricalCrossentropy()
    model = Model(loss=loss, optimizer=optimizer)

    # 2) Add layers to the model
    #   (don't forget to specify the input shape for the first layer)
    model.add(Dense(units=128, input_shape=(784,)))
    model.add(ReLU())

    model.add(Dense(units=64))
    model.add(ReLU())

    model.add(Dense(units=10))
    model.add(Softmax())

    model.fit(x_train=x_train, y_train=y_train, batch_size=10, epochs=10, y_valid=y_valid, x_valid=x_valid)

    return model


# ============================== 3.3.2 convolve ==============================
def convolve(inputs, kernels, padding=0):
    """
    :param inputs: np.array((n, d, ih, iw)), input values
    :param kernels: np.array((c, d, kh, kw)), convolution kernels
    :param padding: int >= 0, the size of padding, 0 means 'valid'

    :return: np.array((n, c, oh, ow)), output values

        n - batch size
        d - number of input channels
        c - number of output channels
        (ih, iw) - input image shape
        (oh, ow) - output image shape
    """
    # !!! Don't change this function, it's here for your reference only !!!
    assert isinstance(padding, int) and padding >= 0
    assert inputs.ndim == 4 and kernels.ndim == 4
    assert inputs.shape[1] == kernels.shape[1]

    if os.environ.get("USE_FAST_CONVOLVE", False):
        return convolve_pytorch(inputs, kernels, padding)
    else:
        return convolve_numpy(inputs, kernels, padding)


def convolve_numpy(inputs, kernels, padding):
    """
    :param inputs: np.array((n, d, ih, iw)), input values
    :param kernels: np.array((c, d, kh, kw)), convolution kernels
    :param padding: int >= 0, the size of padding, 0 means 'valid'

    :return: np.array((n, c, oh, ow)), output values

        n - batch size
        d - number of input channels
        c - number of output channels
        (ih, iw) - input image shape
        (oh, ow) - output image shape
    """
    kernels = np.flip(kernels, axis=(2, 3))

    n, d, ih, iw = inputs.shape
    c, _, kh, kw = kernels.shape

    oh = ih - kh + 2 * padding + 1
    ow = iw - kw + 2 * padding + 1

    output = np.zeros((n, c, oh, ow))
    padded_inputs = np.pad(inputs, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant')

    for out_channel in range(c):
        for in_channel in range(d):
            for i in range(oh):
                for j in range(ow):
                    output[:, out_channel, i, j] += np.sum(
                        padded_inputs[:, in_channel, i:i + kh, j:j + kw] * kernels[out_channel, in_channel, :,
                                                                           :], axis=(-1, -2)
                    )

    return output


# =============================== 4.1.1 Conv2D ===============================
class Conv2D(Layer):
    def __init__(self, output_channels, kernel_size=3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert kernel_size % 2, "Kernel size should be odd"

        self.output_channels = output_channels
        self.kernel_size = kernel_size

        self.kernels, self.kernels_grad = None, None
        self.biases, self.biases_grad = None, None

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        input_channels, input_h, input_w = self.input_shape
        output_channels = self.output_channels
        kernel_size = self.kernel_size

        self.kernels, self.kernels_grad = self.add_parameter(
            name="kernels",
            shape=(output_channels, input_channels, kernel_size, kernel_size),
            initializer=he_initializer(input_h * input_w * input_channels),
        )

        self.biases, self.biases_grad = self.add_parameter(
            name="biases",
            shape=(output_channels,),
            initializer=np.zeros,
        )

        self.output_shape = (output_channels,) + self.input_shape[1:]

    def forward_impl(self, inputs):
        """
        :param inputs: np.array((n, d, h, w)), input values

        :return: np.array((n, c, h, w)), output values

            n - batch size
            d - number of input channels
            c - number of output channels
            (h, w) - image shape
        """
        return convolve(inputs, self.kernels, (self.kernel_size - 1) // 2) + self.biases.reshape(1, -1, 1, 1)

    def backward_impl(self, grad_outputs):
        """
        :param grad_outputs: np.array((n, c, h, w)), dLoss/dOutputs

        :return: np.array((n, d, h, w)), dLoss/dInputs

            n - batch size
            d - number of input channels
            c - number of output channels
            (h, w) - image shape
        """
        self.biases_grad = np.sum(grad_outputs, axis=(0, 2, 3))
        _, _, h, _ = grad_outputs.shape
        p = (self.kernel_size - 1) // 2

        grad_inputs = convolve(
            inputs=grad_outputs,
            kernels=self.kernels[:, :, ::-1, ::-1].transpose((1, 0, 2, 3)),
            padding=self.kernel_size - p - 1
        )

        self.kernels_grad = convolve(
            inputs=self.forward_inputs[:, :, ::-1, ::-1].transpose((1, 0, 2, 3)),
            kernels=grad_outputs.transpose((1, 0, 2, 3)),
            padding=p
        ).transpose(1, 0, 2, 3)

        return grad_inputs


# ============================== 4.1.2 Pooling2D =============================
class Pooling2D(Layer):
    def __init__(self, pool_size=2, pool_mode="max", *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert pool_mode in {"avg", "max"}

        self.pool_size = pool_size
        self.pool_mode = pool_mode
        self.forward_idxs = None

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        channels, input_h, input_w = self.input_shape
        output_h, rem_h = divmod(input_h, self.pool_size)
        output_w, rem_w = divmod(input_w, self.pool_size)
        assert not rem_h, "Input height should be divisible by the pool size"
        assert not rem_w, "Input width should be divisible by the pool size"

        self.output_shape = (channels, output_h, output_w)

    def forward_impl(self, inputs):
        """
        :param inputs: np.array((n, d, ih, iw)), input values

        :return: np.array((n, d, oh, ow)), output values

            n - batch size
            d - number of channels
            (ih, iw) - input image shape
            (oh, ow) - output image shape
        """
        n, d, ih, iw = inputs.shape
        oh = ih // self.pool_size
        ow = iw // self.pool_size
        temp_inputs = inputs.reshape(n, d, oh, self.pool_size, ow, self.pool_size)

        if self.pool_mode == "avg":
            self.forward_idxs = np.full(inputs.shape, 1 / (self.pool_size ** 2))
            return np.mean(temp_inputs, axis=(3, 5))

        elif self.pool_mode == "max":
            temp = np.max(temp_inputs, axis=(3, 5))
            temp_inputs = temp_inputs.transpose(0, 1, 2, 4, 3, 5).reshape(n, d, oh, ow, self.pool_size ** 2)
            zeros = np.zeros_like(temp_inputs)
            np.put_along_axis(zeros, np.argmax(temp_inputs, axis=-1, keepdims=True), 1, axis=-1)
            self.forward_idxs = (zeros
                                 .reshape(n, d, oh, ow, self.pool_size, self.pool_size)
                                 .transpose(0, 1, 2, 4, 3, 5)
                                 .reshape(n, d, ih, iw))
            return temp

    def backward_impl(self, grad_outputs):
        """
        :param grad_outputs: np.array((n, d, oh, ow)), dLoss/dOutputs

        :return: np.array((n, d, ih, iw)), dLoss/dInputs

            n - batch size
            d - number of channels
            (ih, iw) - input image shape
            (oh, ow) - output image shape
        """
        return np.repeat(np.repeat(grad_outputs, self.pool_size, axis=2), self.pool_size, axis=3) * self.forward_idxs


# ============================== 4.1.3 BatchNorm =============================
class BatchNorm(Layer):
    def __init__(self, momentum=0.9, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.momentum = momentum

        self.running_mean = None
        self.running_var = None

        self.beta, self.beta_grad = None, None
        self.gamma, self.gamma_grad = None, None

        self.forward_inverse_std = None
        self.forward_centered_inputs = None
        self.forward_normalized_inputs = None

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        input_channels, input_h, input_w = self.input_shape
        self.running_mean = np.zeros((input_channels,))
        self.running_var = np.ones((input_channels,))

        self.beta, self.beta_grad = self.add_parameter(
            name="beta",
            shape=(input_channels,),
            initializer=np.zeros,
        )

        self.gamma, self.gamma_grad = self.add_parameter(
            name="gamma",
            shape=(input_channels,),
            initializer=np.ones,
        )

    def forward_impl(self, inputs):
        """
        :param inputs: np.array((n, d, h, w)), input values

        :return: np.array((n, d, h, w)), output values

            n - batch size
            d - number of channels
            (h, w) - image shape
        """
        if self.is_training:
            mean_ = np.mean(inputs, axis=(0, 2, 3))
            var_ = np.var(inputs, axis=(0, 2, 3))
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean_
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var_
        else:
            mean_ = self.running_mean
            var_ = self.running_var

        self.forward_inverse_std = (1 / np.sqrt(var_ + eps))[..., np.newaxis, np.newaxis]
        self.forward_centered_inputs = inputs - mean_[..., np.newaxis, np.newaxis]
        self.forward_normalized_inputs = (self.forward_centered_inputs *
                                          self.forward_inverse_std)
        return (self.forward_normalized_inputs * self.gamma[..., np.newaxis, np.newaxis] +
                self.beta[..., np.newaxis, np.newaxis])

    def summ(self, x):
        return np.sum(x, axis=(0, 2, 3))[..., np.newaxis, np.newaxis]

    def backward_impl(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, d, h, w)), dLoss/dOutputs

            :return: np.array((n, d, h, w)), dLoss/dInputs

                n - batch size
                d - number of channels
                (h, w) - image shape
        """

        n, _, h, w = grad_outputs.shape
        coef = 1 / (n * h * w)
        self.gamma_grad = (grad_outputs * self.forward_normalized_inputs).sum(axis=(0, 2, 3))
        self.beta_grad = np.sum(grad_outputs, axis=(0, 2, 3))

        dnorm = grad_outputs * self.gamma[..., np.newaxis, np.newaxis]

        dvar = self.forward_inverse_std ** 2 * self.summ(dnorm * self.forward_centered_inputs)

        db = dvar * coef * (self.forward_centered_inputs - self.summ(self.forward_centered_inputs) * coef)

        return self.forward_inverse_std * (dnorm - self.summ(dnorm) * coef - db)


# =============================== 4.1.4 Flatten ==============================
class Flatten(Layer):
    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        self.output_shape = (int(np.prod(self.input_shape)),)

    def forward_impl(self, inputs):
        """
        :param inputs: np.array((n, d, h, w)), input values

        :return: np.array((n, (d * h * w))), output values

            n - batch size
            d - number of input channels
            (h, w) - image shape
        """
        n, d, h, w = inputs.shape
        return inputs.reshape(n, d * h * w)

    def backward_impl(self, grad_outputs):
        """
        :param grad_outputs: np.array((n, (d * h * w))), dLoss/dOutputs

        :return: np.array((n, d, h, w)), dLoss/dInputs

            n - batch size
            d - number of units
            (h, w) - input image shape
        """
        n, _ = grad_outputs.shape
        d, h, w = self.input_shape
        return grad_outputs.reshape(n, d, h, w)


# =============================== 4.1.5 Dropout ==============================
class Dropout(Layer):
    def __init__(self, p, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p = p
        self.forward_mask = None

    def forward_impl(self, inputs):
        """
        :param inputs: np.array((n, ...)), input values

        :return: np.array((n, ...)), output values

            n - batch size
            ... - arbitrary shape (the same for input and output)
        """
        if self.is_training:
            rasp = np.random.uniform(0, 1, inputs.size)
            self.forward_mask = rasp.reshape(*inputs.shape) > self.p
            return self.forward_mask * inputs
        else:
            return (1 - self.p) * inputs

    def backward_impl(self, grad_outputs):
        """
        :param grad_outputs: np.array((n, ...)), dLoss/dOutputs

        :return: np.array((n, ...)), dLoss/dInputs

            n - batch size
            ... - arbitrary shape (the same for input and output)
        """
        return self.forward_mask * grad_outputs


# ====================== 2.3 Train and Test on CIFAR-10 ======================
def train_cifar10_model(x_train, y_train, x_valid, y_valid):
    optimizer = SGDMomentum(lr=0.01, momentum=0.9)
    loss = CategoricalCrossentropy()
    model = Model(loss=loss, optimizer=optimizer)

    model.add(Conv2D(8, 3, input_shape=(3, 32, 32)))
    model.add(ReLU())
    model.add(BatchNorm())
    model.add(Pooling2D(2, 'max'))
    model.add(Dropout(0.2))

    model.add(Conv2D(16, 3))
    model.add(ReLU())
    model.add(BatchNorm())
    model.add(Pooling2D(2, 'max'))
    model.add(Dropout(0.2))

    model.add(Conv2D(32, 3))
    model.add(ReLU())
    model.add(BatchNorm())
    model.add(Pooling2D(2, 'max'))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(ReLU())
    model.add(Dense(10))
    model.add(Softmax())

    model.fit(x_train, y_train, batch_size=32, epochs=10, x_valid=x_valid, y_valid=y_valid)

    return model

# ============================================================================
