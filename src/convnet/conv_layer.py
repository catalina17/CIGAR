from layer import Layer
import numpy


class ConvLayer(Layer):

    def __init__(self, num_filters, filter_shape, weight_decay):
        self.num_filters = num_filters
        self.filter_shape = filter_shape
        self.filter_weights = numpy.random.rand(filter_shape[0], filter_shape[1])

        self.weight_adjustments = numpy.zeros(filter_shape)
        self.weight_decay = weight_decay

        self.input_shape = None
        self.output_shape = None

    def forward_prop(self, input):
        raise NotImplementedError()

    def back_prop(self, output_grad):
        raise NotImplementedError()

    def set_input_shape(self, shape):
        self.input_shape = shape

    def get_output_shape(self):
        return self.output_shape

    def update_parameters(self, learning_rate):
        raise NotImplementedError()
