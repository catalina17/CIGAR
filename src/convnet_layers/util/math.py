import numpy as np


def sigmoid(x):
    """
    sigmoid(x) = 1 / (1 + e^(-x))

    Parameters
    ----------
    x : scalar or array

    Returns
    -------
    scalar or array
        The result of applying the sigmoid function to each element of the input.

    """
    return 1.0 / (1.0 + np.exp(-x))


def d_sigmoid(x):
    """
    sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))

    Parameters
    ----------
    x : scalar or array

    Returns
    -------
    scalar or array
        The result of applying the derivative of the sigmoid function to each element of the input.

    """
    s = sigmoid(x)
    return s * (1 - s)


def relu(x):
    """
    ReLU(x) = max(0, x)

    Parameters
    ----------
    x : scalar or array

    Returns
    -------
    scalar or array
        The result of applying the ReLU function to each element of the input.

    """
    return np.maximum(0, x)


def d_relu(x):
    """
    ReLU'(x) = { 1, if x > 0
               { 0, otherwise

    Parameters
    ----------
    x : scalar or array

    Returns
    -------
    scalar or array
        The result of applying the derivative of the ReLU function to each element of the input.

    """
    res = np.zeros(x.shape)
    res[x > 0] = 1

    return res


def leaky_relu(x):
    """
    leakyReLU(x) = { x       , if x > 0
                   { 0.01 * x, otherwise

    Parameters
    ----------
    x : scalar or array

    Returns
    -------
    scalar or array
        The result of applying the leakyReLU function to each element of the input.

    """
    res = x
    res[x <= 0] *= 0.01

    return res


def d_leaky_relu(x):
    """
    leakyReLU'(x) = { 1   , if x > 0
                    { 0.01, otherwise

    Parameters
    ----------
    x : scalar or array

    Returns
    -------
    scalar or array
        The result of applying the derivative of the leakyReLU function to each element of the
        input.

    """
    res = np.zeros(x.shape)
    res[x > 0] = 1
    res[x <= 0] = 0.01

    return res
