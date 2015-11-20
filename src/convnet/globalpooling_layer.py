from layer import Layer


class GlobalPoolingLayer(Layer):

    def __init__(self, filter_size, stride):
        raise NotImplementedError()

    def forward_prop(self, input):
        raise NotImplementedError()

    def back_prop(self, output_grad):
        raise NotImplementedError()

    def _input_size(self):
        raise NotImplementedError()

    def _output_size(self):
        raise NotImplementedError()
