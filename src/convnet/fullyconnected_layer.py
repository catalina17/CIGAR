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
        self.d_biases = np.empty(num_nodes)

        self.input_shape = None
        self.current_input = None

        self.weights = None
        self.d_weights = None

    def forward_prop(self, input):
        assert self.input_shape == input.shape, "Input does not have correct shape"
        self.current_input = input
        return np.dot(input, self.weights) + self.biases

    def back_prop(self, output_grad):
        self.d_weights = np.dot(np.resize(self.current_input, (self.input_shape[0], 1)),
                                np.resize(output_grad, (output_grad.shape[0], 1)).T)

        print "Weights derivative: "
        print self.d_weights
        self.d_biases = output_grad
        print "Biases derivative: "
        print self.d_biases

        print "Propagated gradient: "
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
        print "Weights: "
        print self.weights

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
        self.weights -= learning_rate * self.d_weights
        self.biases -= learning_rate * self.d_biases

if __name__ == "__main__":
    dummy_input = np.ones((4,))
    print "Input: "
    print dummy_input

    layer = FullyConnectedLayer(5)
    layer.set_input_shape((4,))

    print "\n--->> Forward propagation:"
    print sum(layer.weights)
    print layer.forward_prop(dummy_input)

    dummy_output_grad = np.random.randn(5)
    print "\nOutput gradient: "
    print dummy_output_grad

    print "\n--->> Backpropagation: "
    print layer.back_prop(dummy_output_grad)

    print "\n--->> Params before update: "
    print layer.weights
    print layer.biases
    layer.update_parameters(0.01)
    print "\n--->> Params after update: "
    print layer.weights
    print layer.biases
