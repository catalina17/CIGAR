import numpy as np
import time

from layer import Layer


class GlobalPoolingLayer(Layer):

    def __init__(self):
        self._input_shape = None
        self._current_input = None
        self._max_activation_indices = None

    def forward_prop(self, input):
        """

        Parameters
        ----------
        input : array of double
            The input for the layer.

        Returns
        -------
        array of double
            The result of the global pooling layer processing the input.

        """
        assert self._input_shape == input.shape, "Input does not have correct shape"
        self._current_input = input

        output = np.empty(self.get_output_shape())

        for f in range(self._input_shape[0]):
            # Average pooling
            output[f] = np.mean(input[f])
            # Max pooling
            output[self._input_shape[0] + f] = np.max(input[f])
            self._max_activation_indices[f] = np.argmax(input[f])
            # L2-norm pooling
            output[2 * self._input_shape[0] + f] = np.sqrt(np.sum(input[f] ** 2))
        return output

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
        # Separate the three types of gradient values to use for computing the new gradient
        mean_output_grad = output_grad[0:self._input_shape[0]]
        max_output_grad = output_grad[self._input_shape[0]:2 * self._input_shape[0]]
        l2_output_grad = output_grad[2 * self._input_shape[0]:]

        input_grad = np.empty(self._input_shape, dtype=np.float64)
        range_f = self._input_shape[0]
        for f in range(range_f):
            # Average grad
            input_grad[f] = mean_output_grad[f] / self._input_shape[1]
            # Send the max gradient value to the location from where the maximum value was
            # extracted during forward propagation
            input_grad[f, self._max_activation_indices[f]] += max_output_grad[f]
            # L2-norm grad
            sqr_root = np.sqrt(np.sum(self._current_input[f] ** 2))
            # Avoid producing exploding values, if square root is very small
            if sqr_root >= 1e-5:
                input_grad[f] += self._current_input[f] / sqr_root * l2_output_grad[f]

        return input_grad

    def set_input_shape(self, shape):
        """

        Parameters
        ----------
        shape : tuple
            The shape of the inputs which this layer will process.

        """
        self._input_shape = shape
        self._max_activation_indices = np.empty(self._input_shape[0], dtype=np.int32)

    def get_output_shape(self):
        """

        Returns
        -------
        tuple
            The output shape of this layer.

        """
        shape = (3 * self._input_shape[0],)
        return shape
