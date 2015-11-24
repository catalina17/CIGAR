from layer import Layer


class SoftmaxLayer(Layer):

    def forward_prop(self, input):
        raise NotImplementedError()

    def back_prop(self, output_grad):
        raise NotImplementedError("Output layer - does not perform backpropagation")

    def get_gradient(self, predicted_output, example_output):
        raise NotImplementedError()

    def set_input_shape(self, shape):
        raise NotImplementedError()

    def get_output_shape(self):
        raise NotImplementedError()

    def update_parameters(self, learning_rate):
        raise NotImplementedError("Output layer - does not perform parameter update")
