import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def d_sigmoid(x):
    s = sigmoid(x)
    return s * (1 - s)


def relu(x):
    return np.maximum(0, x)


def d_relu(x):
    res = np.zeros(x.shape)
    res[x > 0] = 1

    return res


def leaky_relu(x):
    res = x
    res[x <= 0] *= 0.01

    return res


def d_leaky_relu(x):
    res = np.zeros(x.shape)
    res[x > 0] = 1
    res[x <= 0] = 0.01

    return res
