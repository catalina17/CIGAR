from layer import Layer
import numpy as np


class GlobalPoolingLayer(Layer):

    def __init__(self):
        self.input_shape = None
        self.current_input = None

    def forward_prop(self, input):
        assert self.input_shape == input.shape, "Input does not have correct shape"
        self.current_input = input

        output = np.empty(self.get_output_shape())
        for filter in self.input_shape[0]:
            # Mean pooling
            output[filter] = np.mean(input[filter])
            # Max pooling
            output[self.input_shape[0] + filter] = np.max(input[filter])
            # L2 pooling
            output[2 * self.input_shape[0] + filter] = np.sqrt(sum(input[filter] ** 2))
        return output

    def back_prop(self, output_grad):
        raise NotImplementedError()

    def set_input_shape(self, shape):
        """

        Parameters
        ----------
        shape : tuple

        """
        self.input_shape = shape
        print "GlobalPoolingLayer with input shape " + str(shape)

    def get_output_shape(self):
        """

        Returns
        -------
        tuple

        """
        shape = (3 * self.input_shape[0], )
        print "GlobalPoolingLayer with output shape " + str(shape)
        return shape
