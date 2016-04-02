import numpy as np
import time

from layer import Layer


class FullyConnectedLayer(Layer):

    def __init__(self, num_nodes, weight_scale):
        """

        Parameters
        ----------
        num_nodes : int
            The number of nodes (outputs) of this layer.
        weight_scale : float
            The standard deviation sigma of the normal distribution N(0, sigma^2) from which
            the filter weights are initialised.

        """
        self._num_nodes = num_nodes
        # Set initial bias values to 0
        self._biases = np.zeros(num_nodes).astype(np.float64)
        self._d_biases = np.zeros(num_nodes).astype(np.float64)

        self._weight_scale = weight_scale

        self._input_shape = None
        self._current_input = None

        self._weights = None
        self._d_weights = None

    def forward_prop(self, input):
        """

        Parameters
        ----------
        input : array of double
            The input for the layer.

        Returns
        -------
        array of double
            The result of the fully-connected layer processing the input.

        """
        assert self._input_shape == input.shape, "Input does not have correct shape"
        self._current_input = input
        return np.dot(input, self._weights) + self._biases

    def back_prop(self, output_grad):
        """

        Parameters
        ----------
        output_grad : array of double
            The incoming gradient from the next layer of the network.

        Returns
        -------
        array of double
            The gradient computed by this layer.

        """
        # Compute derivatives for weight matrix and bias values
        self._d_weights += np.dot(np.resize(self._current_input, (self._input_shape[0], 1)),
                                  np.resize(output_grad, (output_grad.shape[0], 1)).T)
        self._d_biases += output_grad

        # Compute new gradient
        return np.dot(output_grad, self._weights.T)

    def set_input_shape(self, shape):
        """

        Parameters
        ----------
        shape : tuple
            The shape of the inputs which this layer will process.

        """
        self._input_shape = shape
        # Initialise weights as described in the __init__ method docstring
        self._weights = np.random.normal(loc=0, scale=self._weight_scale,
                                         size=(shape[0], self._num_nodes)).astype(np.float64)
        self._d_weights = np.zeros(self._weights.shape).astype(np.float64)

    def get_output_shape(self):
        """

        Returns
        -------
        tuple
            The output shape of this layer.

        """
        shape = (self._num_nodes,)
        return shape

    def update_parameters(self, learning_rate):
        """

        Parameters
        ----------
        learning_rate : float
            The learning rate used to update the filter weights and biases.

        """
        # Update weight matrix and bias values with derivatives computed during back-propagation
        self._weights -= learning_rate * self._d_weights
        self._biases -= learning_rate * self._d_biases

        self._d_weights[...] = 0
        self._d_biases[...] = 0

    def serialise_parameters(self, file_idx):
        """

        Parameters
        ----------
        file_idx : int
            The index associated with this layer in the network structure for which we wish to save
            parameters.

        """
        param_file = open('saved_params/FC_' + str(file_idx) + '_weights', 'w')
        np.save(param_file, self._weights)
        param_file.close()

        param_file = open('saved_params/FC_' + str(file_idx) + '_biases', 'w')
        np.save(param_file, self._biases)
        param_file.close()

    def init_parameters_from_file(self, file_idx):
        """

        Parameters
        ----------
        file_idx : int
            The index associated with this layer in the network structure for which we wish to
            retrieve the saved parameters.

        """
        param_file = open('saved_params/FC_' + str(file_idx) + '_weights', 'rb')
        self._weights = np.load(param_file)
        param_file.close()

        param_file = open('saved_params/FC_' + str(file_idx) + '_biases', 'rb')
        self._biases = np.load(param_file)
        param_file.close()
