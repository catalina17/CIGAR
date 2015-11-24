from layer import Layer


class MaxPoolingLayer(Layer):

    def __init__(self, filter_size, stride):
        raise NotImplementedError()

    def forward_prop(self, input):
        raise NotImplementedError()

    def back_prop(self, output_grad):
        raise NotImplementedError()

    def set_input_shape(self, shape):
        raise NotImplementedError()

    def get_output_shape(self):
        raise NotImplementedError()

    def update_parameters(self, learning_rate):
        raise NotImplementedError()
