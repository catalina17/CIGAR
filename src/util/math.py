import numpy as np
import numpy.core.umath as umath


def sigmoid(x):
    return 1.0 / (1.0 + umath.exp(-x))


def d_sigmoid(x):
    s = sigmoid(x)
    return s * (1 - s)


def relu(x):
    return umath.maximum(x, np.zeros(x.size))


def d_relu(x):
    res = np.zeros(x.size)
    res[x >= 0] = 1

    return res
