import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def d_sigmoid(x):
    s = sigmoid(x)
    return s * (1 - s)


def relu(x):
    return np.log(1 + np.exp(x))


def d_relu(x):
    return sigmoid(x)
