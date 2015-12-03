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
        self.current_input = None
        self.weights = None

    def forward_prop(self, input):
        assert self.input_shape == input.shape, "Input does not have correct shape"
        self.current_input = input
        return np.dot(input, self.weights) + self.biases

    def back_prop(self, output_grad):
        return np.dot(output_grad, self.weights.T)

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
    dummy_input = np.ones((4,))
    print "Input: "
    print dummy_input

    layer = FullyConnectedLayer(5)
    layer.set_input_shape((4,))

    print "Forward propagation:"
    print sum(layer.weights)
    print layer.forward_prop(dummy_input)
    print '\n'

    dummy_output_grad = np.ones((5,))
    print "Output gradient: "
    print dummy_output_grad

    print "Backpropagation: "
    print sum(layer.weights.T)
    print layer.back_prop(dummy_output_grad)
