from layer import Layer


class FullyConnectedLayer(Layer):

    def __init__(self, num_nodes):
        raise NotImplementedError()

    def forward_prop(self, input):
        raise NotImplementedError()

    def back_prop(self, output_grad):
        raise NotImplementedError()

    def _input_size(self):
        raise NotImplementedError()

    def _output_size(self):
        raise NotImplementedError()
