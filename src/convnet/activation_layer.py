import numpy as np
import time

from layer import Layer
from util import math


class ActivationLayer(Layer):

    def __init__(self, activation_fn):
        if activation_fn == 'ReLU':
            self._activation_fn = math.relu
            self._d_activation_fn = math.d_relu
        elif activation_fn == 'leakyReLU':
            self._activation_fn = math.leaky_relu
            self._d_activation_fn = math.d_leaky_relu
        elif activation_fn == 'sigmoid':
            self._activation_fn = math.sigmoid
            self._d_activation_fn = math.d_sigmoid
        elif activation_fn == 'softplus':
            self._activation_fn = math.softplus
            self._d_activation_fn = math.d_softplus

        self._input_shape = None
        self._current_input = None

    def forward_prop(self, input):
        assert self._input_shape == input.shape, "Input does not have correct shape"
        self._current_input = input
        return self._activation_fn(input)

    def back_prop(self, output_grad):
        return output_grad * self._d_activation_fn(self._current_input)

    def set_input_shape(self, shape):
        self._input_shape = shape

    def get_output_shape(self):
        return self._input_shape

if __name__ == '__main__':
    dummy_input = np.random.randn(64, 596)
    print "Input:\n", dummy_input

    layer = ActivationLayer('leakyReLU')
    layer.set_input_shape((64, 596))

    start = time.time()
    print "\n--->> Forward propagation:\n", layer.forward_prop(dummy_input)
    finish = time.time()
    print "Time taken: ", finish - start

    dummy_output_grad = np.ones((64, 596))
    print "Output gradient:\n", dummy_output_grad

    start = time.time()
    print "\n--->> Backpropagation:\n", layer.back_prop(dummy_output_grad)
    finish = time.time()
    print "Time taken: ", finish - start
