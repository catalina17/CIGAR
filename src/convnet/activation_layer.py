from layer import Layer
from util import math


class ActivationLayer(Layer):

    def __init__(self, activation_fn):
        if activation_fn == 'ReLU':
            self.activation_fn = math.relu
            self.d_activation_fn = math.d_relu

        self.input_shape = None
        self.current_input = None

    def forward_prop(self, input):
        self.current_input = input
        return self.activation_fn(input)

    def back_prop(self, output_grad):
        return output_grad * self.d_activation_fn(self.current_input)

    def set_input_shape(self, shape):
        self.input_shape = shape
        print "ActivationLayer with input shape " + str(shape)

    def get_output_shape(self):
        print "ActivationLayer with output shape " + str(self.input_shape)
        return self.input_shape

    def update_parameters(self, learning_rate):
        raise NotImplementedError()
