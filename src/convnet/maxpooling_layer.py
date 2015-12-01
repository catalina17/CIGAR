from layer import Layer
import numpy as np


class MaxPoolingLayer(Layer):
    def __init__(self, filter_shape):
        """

        Parameters
        ----------
        filter_shape : tuple

        """
        self.filter_shape = filter_shape
        print "Filter shape: " + str(filter_shape)
        self.input_shape = None

    def forward_prop(self, input):
        assert self.input_shape == input.shape, "Input does not have correct shape"

        if (len(self.get_output_shape()) == 1):
            output = np.zeros((1, self.get_output_shape()[0]))
        else:
            output = np.zeros(self.get_output_shape())

        filter_h = self.filter_shape[0]
        filter_w = self.filter_shape[1]
        input_h = self.input_shape[0]
        input_w = self.input_shape[1]

        range_i = input_h/filter_h
        range_j = input_w/filter_w

        for i in range(range_i):
            for j in range(range_j):
                current_h = i * filter_h
                current_w = j * filter_w

                output[i, j] = input[current_h:current_h + filter_h,
                                     current_w:current_w + filter_w].max(axis=0).max(axis=0)

        if output.shape[0] == 1:
            return output[0]
        else:
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
        print "MaxPoolingLayer with input shape " + str(shape)

    def get_output_shape(self):
        """

        Returns
        -------
        tuple

        """
        assert self.input_shape[0] % self.filter_shape[0] == 0 and\
            self.input_shape[1] % self.filter_shape[1] == 0,\
            "Input shape is not a multiple of filter shape in MaxPoolingLayer"

        shape = None

        if self.input_shape[0] / self.filter_shape[0] == 1:
            shape = (self.input_shape[1] / self.filter_shape[1], )
        else:
            shape = (self.input_shape[0] / self.filter_shape[0],
                     self.input_shape[1] / self.filter_shape[1])

        print "MaxPoolingLayer with output shape " + str(shape)
        return shape

    def update_parameters(self, learning_rate):
        """

        Parameters
        ----------
        learning_rate : float

        """
        raise NotImplementedError()

if __name__ == "__main__":
    dummy_input = np.random.randn(2, 4)
    print dummy_input

    maxpool_layer = MaxPoolingLayer(filter_shape=(2,2))
    maxpool_layer.set_input_shape(dummy_input.shape)

    res = maxpool_layer.forward_prop(dummy_input)
    print res
