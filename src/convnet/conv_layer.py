from layer import Layer
import numpy as np


class ConvLayer(Layer):

    def __init__(self, num_filters, filter_shape, weight_decay, padding_mode=True):
        """

        Parameters
        ----------
        num_filters : int
        filter_shape : tuple
        weight_decay : float
        padding_mode : bool

        """
        self.num_filters = num_filters
        self.filter_shape = filter_shape

        self.filter_weights = np.ndarray(num_filters, dtype=np.ndarray)
        for i in range(num_filters):
            self.filter_weights[i] = np.random.rand(filter_shape[0], filter_shape[1])
        print "Weights: "
        print self.filter_weights

        self.d_weights = [np.zeros(filter_shape)]
        self.weight_decay = weight_decay

        if padding_mode:
            # Padding along height dimension
            self.num_padding_zeros = 2 * (filter_shape[1] - 1)
        else:
            self.num_padding_zeros = 0

        self.input_shape = None

    def forward_prop(self, input):
        assert self.input_shape == input.shape, "Input does not have correct shape"

        padded_input = np.zeros((self.input_shape[0], self.input_shape[1] + self.num_padding_zeros))
        padded_input[0:self.input_shape[0],
                     self.num_padding_zeros / 2:
                     self.num_padding_zeros / 2 + self.input_shape[1]] = input

        output = np.zeros(self.get_output_shape())

        filter_h = self.filter_shape[0]
        filter_w = self.filter_shape[1]
        range_i = self.get_output_shape()[1] - filter_w + 1

        for f in range(self.num_filters):
            for i in range(range_i):
                output[f][i] = sum(sum(np.multiply(self.filter_weights[f],
                                                   padded_input[0:filter_h, i:i+filter_w])))

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
        print "ConvLayer with input shape " + str(shape)

    def get_output_shape(self):
        """

        Returns
        -------
        tuple

        """
        shape = (self.num_filters,
                 self.input_shape[1] + self.num_padding_zeros - self.filter_shape[1] + 1)
        print "ConvLayer with output shape " + str(shape)
        return shape

    def update_parameters(self, learning_rate):
        raise NotImplementedError()

if __name__ == "__main__":

    layer = ConvLayer(5, (4, 4), 0.01, True)
    layer.set_input_shape((8, 8))

    dummy_input = np.random.randn(8, 8)
    print "Dummy input: "
    print dummy_input

    res = layer.forward_prop(dummy_input)
    print "Output of Conv layer: " + str(res.shape)
    print res
