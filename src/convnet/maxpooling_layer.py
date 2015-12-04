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
        self.max_activation_indices = None

    def forward_prop(self, input):
        assert self.input_shape == input.shape, "Input does not have correct shape"

        if (len(self.get_output_shape()) == 1):
            output = np.empty((1, self.get_output_shape()[0]))
        else:
            output = np.empty(self.get_output_shape())

        self.max_activation_indices = np.empty(self.get_output_shape() + (2,))

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

                max_val = -np.inf
                h_max = 0
                w_max = 0
                for h in range(current_h, current_h + filter_h):
                    for w in range(current_w, current_w + filter_w):
                        if input[h, w] > max_val:
                            h_max = h
                            w_max = w
                            max_val = input[h, w]
                output[i, j] = max_val
                self.max_activation_indices[i, j, 0] = h_max
                self.max_activation_indices[i, j, 1] = w_max

        if output.shape[0] == 1:
            return output[0]
        else:
            return output

    def back_prop(self, output_grad):
        input_grad = np.zeros(self.input_shape)

        for i in range(output_grad.shape[0]):
            for j in range(output_grad.shape[1]):
                h = self.max_activation_indices[i, j, 0]
                w = self.max_activation_indices[i, j, 1]
                input_grad[h][w] = output_grad[i][j]

        return input_grad

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

if __name__ == "__main__":
    dummy_input = np.random.randn(4, 4)
    print "Input: "
    print dummy_input

    layer = MaxPoolingLayer(filter_shape=(2,2))
    layer.set_input_shape(dummy_input.shape)

    print "Forward propagation:"
    res = layer.forward_prop(dummy_input)
    print res
    print '\n'

    dummy_output_grad = np.ones((2, 2))
    print "Output gradient: "
    print dummy_output_grad

    print "Backpropagation:"
    print layer.back_prop(dummy_output_grad)
