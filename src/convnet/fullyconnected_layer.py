from layer import Layer
import numpy as np


class FullyConnectedLayer(Layer):

    def __init__(self, num_nodes, weight_decay, weight_scale):
        """

        Parameters
        ----------
        num_nodes : int
        weight_decay : float
        weight_scale : float

        """
        self.num_nodes = num_nodes
        self.biases = np.zeros(num_nodes)
        self.d_biases = np.zeros(num_nodes)

        self.weight_decay = weight_decay
        self.weight_scale = weight_scale

        self.input_shape = None
        self.current_input = None

        self.weights = None
        self.d_weights = None

    def forward_prop(self, input):
        assert self.input_shape == input.shape, "Input does not have correct shape"
        self.current_input = input
        return np.dot(input, self.weights) + self.biases

    def back_prop(self, output_grad):
        self.d_weights += np.dot(np.resize(self.current_input, (self.input_shape[0], 1)),
                                 np.resize(output_grad, (output_grad.shape[0], 1)).T) +\
                          self.weight_decay * self.weights
        # print "Weights derivative ex:\n", self.d_weights[0][0]

        self.d_biases += output_grad
        # print "Biases derivative ex:\n", self.d_biases[0]

        return np.dot(output_grad, self.weights.T)

    def set_input_shape(self, shape):
        """

        Parameters
        ----------
        shape : tuple

        """
        self.input_shape = shape
        self.weights = np.random.normal(loc=0, scale=self.weight_scale,
                                        size=(shape[0], self.num_nodes))
        self.d_weights = np.zeros(self.weights.shape)

        # print "FullyConnectedLayer with input shape " + str(shape)

    def get_output_shape(self):
        """

        Returns
        -------
        tuple

        """
        shape = (self.num_nodes, )
        # print "FullyConnectedLayer with output shape " + str(shape)
        return shape

    def update_parameters(self, learning_rate):
        """

        Parameters
        ----------
        learning_rate : float

        """
        self.weights -= learning_rate * self.d_weights
        self.biases -= learning_rate * self.d_biases

        self.d_weights[...] = 0
        self.d_biases[...] = 0

if __name__ == "__main__":
    dummy_input = np.ones((4,))
    print "Input:\n", dummy_input

    layer = FullyConnectedLayer(5, weight_decay=0.1, weight_scale=0.01)
    layer.set_input_shape((4,))

    print "\n--->> Forward propagation:\n", sum(layer.weights), "\n", layer.forward_prop(dummy_input)

    dummy_output_grad = np.random.randn(5)
    print "\nOutput gradient:\n", dummy_output_grad

    print "\n--->> Backpropagation:\n", layer.back_prop(dummy_output_grad)

    print "\n--->> Params before update:\n", layer.weights, "\n", layer.biases
    layer.update_parameters(0.01)
    print "\n--->> Params after update:\n", layer.weights, "\n", layer.biases
