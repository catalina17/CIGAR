from layer import Layer


class SoftmaxLayer(Layer):

    def __init__(self):
        self.n_nodes = None

    def forward_prop(self, input):
        raise NotImplementedError()

    def back_prop(self, output_grad):
        raise NotImplementedError("Output (Softmax) layer - does not perform backpropagation")

    def get_gradient(self, predicted_output, example_output):
        raise NotImplementedError()

    def set_input_shape(self, shape):
        """

        Parameters
        ----------
        shape : tuple

        """
        assert len(shape) == 1, "Input shape not suitable for output (Softmax) layer"
        self.n_nodes = shape[0]
        print "SoftmaxLayer with input shape " + shape

    def get_output_shape(self):
        """

        Returns
        -------
        tuple

        """
        shape = (self.n_nodes, )
        print "SoftmaxLayer with output shape " + shape
        return shape

    def update_parameters(self, learning_rate):
        raise NotImplementedError("Output layer - does not perform parameter update")
