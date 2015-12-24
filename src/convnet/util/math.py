import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def d_sigmoid(x):
    s = sigmoid(x)
    return s * (1 - s)


def softplus(x):
    return np.log(1 + np.exp(x))


def d_softplus(x):
    return softplus(x)


def relu(x):
    return np.maximum(0, x)


def d_relu(x):
    res = np.zeros(x.shape)
    res[x >= 0] = 1

    return res
