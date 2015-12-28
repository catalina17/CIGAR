from layer import Layer
from util import math
import numpy as np


class ActivationLayer(Layer):

    def __init__(self, activation_fn):
        if activation_fn == 'ReLU':
            self.activation_fn = math.relu
            self.d_activation_fn = math.d_relu
        elif activation_fn == 'leakyReLU':
            self.activation_fn = math.leaky_relu
            self.d_activation_fn = math.d_leaky_relu
        elif activation_fn == 'sigmoid':
            self.activation_fn = math.sigmoid
            self.d_activation_fn = math.d_sigmoid
        elif activation_fn == 'softplus':
            self.activation_fn = math.softplus
            self.d_activation_fn = math.d_softplus

        self.input_shape = None
        self.current_input = None

    def forward_prop(self, input):
        assert self.input_shape == input.shape, "Input does not have correct shape"
        self.current_input = input
        return self.activation_fn(input)

    def back_prop(self, output_grad):
        return output_grad * self.d_activation_fn(self.current_input)

    def set_input_shape(self, shape):
        self.input_shape = shape
        # print "ActivationLayer with input shape " + str(shape) + " and activation function " +
        # str(self.activation_fn.__name__)

    def get_output_shape(self):
        # print "ActivationLayer with output shape " + str(self.input_shape)
        return self.input_shape

if __name__ == '__main__':
    dummy_input = np.random.randn(4, 4)
    print "Input:\n", dummy_input

    layer = ActivationLayer('ReLU')
    layer.set_input_shape((4, 4))

    print "\n--->> Forward propagation:\n", layer.forward_prop(dummy_input)

    dummy_output_grad = np.ones((4, 4))
    print "Output gradient:\n", dummy_output_grad

    print "\n--->> Backpropagation:\n", layer.back_prop(dummy_output_grad)
