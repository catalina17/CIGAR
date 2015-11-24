from layer import Layer


class FullyConnectedLayer(Layer):

    def __init__(self, num_nodes):
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
