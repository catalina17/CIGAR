from layer import Layer
from util import math


class ActivationLayer(Layer):

    def __init__(self, activation_fn):
        """

        Parameters
        ----------
        activation_fn : str
            The name of the activation function for this layer.

        """
        if activation_fn == 'ReLU':
            self._activation_fn = math.relu
            self._d_activation_fn = math.d_relu
        elif activation_fn == 'leakyReLU':
            self._activation_fn = math.leaky_relu
            self._d_activation_fn = math.d_leaky_relu
        elif activation_fn == 'sigmoid':
            self._activation_fn = math.sigmoid
            self._d_activation_fn = math.d_sigmoid

        self._input_shape = None
        self._current_input = None

    def forward_prop(self, input):
        """

        Parameters
        ----------
        input : array of double
            The input for the layer.

        Returns
        -------
        array of double
            The result of the layer applying the activation function to the input.

        """
        assert self._input_shape == input.shape, "Input does not have correct shape"
        self._current_input = input
        return self._activation_fn(input)

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
        return output_grad * self._d_activation_fn(self._current_input)

    def set_input_shape(self, shape):
        """

        Parameters
        ----------
        shape : tuple
            The shape of the inputs which this layer will process.

        """
        self._input_shape = shape

    def get_output_shape(self):
        """

        Returns
        -------
        tuple
            The output shape of this layer.

        """
        return self._input_shape
