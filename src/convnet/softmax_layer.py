from layer import Layer
import numpy as np


class SoftmaxLayer(Layer):

    def __init__(self):
        self.num_nodes = None

    def forward_prop(self, input):
        input -= np.max(input)
        output = np.exp(input) / np.sum(np.exp(input))

        return output

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
        self.num_nodes = shape[0]
        print "SoftmaxLayer with input shape " + str(shape)

    def get_output_shape(self):
        """

        Returns
        -------
        tuple

        """
        shape = (self.num_nodes, )
        print "SoftmaxLayer with output shape " + str(shape)
        return shape

    def update_parameters(self, learning_rate):
        raise NotImplementedError("Output layer - does not perform parameter update")

if __name__ == "__main__":
    dummy_input = np.zeros(3)
    dummy_input[0] = 0
    dummy_input[1] = 1
    dummy_input[2] = 2
    print dummy_input

    layer = SoftmaxLayer()
    layer.set_input_shape((3,))

    res = layer.forward_prop(dummy_input)
    print res
