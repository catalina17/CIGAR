from layer import Layer
import numpy as np


class FullyConnectedLayer(Layer):

    def __init__(self, num_nodes):
        """

        Parameters
        ----------
        num_nodes : int

        """
        self.num_nodes = num_nodes
        self.biases = np.zeros(num_nodes)
        self.input_shape = None
        self.weights = None

    def forward_prop(self, input):
        assert self.input_shape == input.shape, "Input does not have correct shape"
        return np.dot(input, self.weights) + self.biases

    def back_prop(self, output_grad):
        raise NotImplementedError()

    def set_input_shape(self, shape):
        """

        Parameters
        ----------
        shape : tuple

        """
        self.input_shape = shape
        self.weights = np.random.randn(shape[0], self.num_nodes)
        print "FullyConnectedLayer with input shape " + str(shape)

    def get_output_shape(self):
        """

        Returns
        -------
        tuple

        """
        shape = (self.num_nodes, )
        print "FullyConnectedLayer with output shape " + str(shape)
        return shape

    def update_parameters(self, learning_rate):
        """

        Parameters
        ----------
        learning_rate : float

        """
        raise NotImplementedError()

if __name__ == "__main__":
    dummy_input = np.random.randn(4)
    print dummy_input

    layer = FullyConnectedLayer(5)
    layer.set_input_shape((4,))

    res = layer.forward_prop(dummy_input)
    print res
