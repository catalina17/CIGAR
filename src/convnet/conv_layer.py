from layer import Layer


class ConvLayer(Layer):

    def __init__(self, num_filters, filter_size, weight_decay):
        raise NotImplementedError()

    def forward_prop(self, input):
        raise NotImplementedError()

    def back_prop(self, output_grad):
        raise NotImplementedError()

    def input_size(self):
        raise NotImplementedError()

    def output_size(self):
        raise NotImplementedError()
