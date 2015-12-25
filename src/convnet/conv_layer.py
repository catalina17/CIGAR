from layer import Layer
import numpy as np


class ConvLayer(Layer):

    def __init__(self, num_filters, filter_shape, weight_decay, weight_scale, padding_mode=True):
        """

        Parameters
        ----------
        num_filters : int
        filter_shape : tuple
        weight_decay : float
        weight_scale : float
        padding_mode : bool

        """
        self.num_filters = num_filters
        self.filter_shape = filter_shape
        print "Filter shape: " + str(filter_shape)

        self.filter_weights = np.empty((num_filters, filter_shape[0], filter_shape[1]))
        for i in range(num_filters):
            self.filter_weights[i] = np.random.normal(loc=0, scale=weight_scale, size=filter_shape)
            # print "Filter weights for filter " + str(i + 1) + ":\n", self.filter_weights[i]
        self.d_filter_weights = np.zeros(self.filter_weights.shape)

        self.biases = np.random.uniform(0.1, 0.2, [num_filters])
        self.d_biases = np.zeros(num_filters)

        self.weight_decay = weight_decay

        if padding_mode:
            # Padding input with columns
            self.num_padding_zeros = 2 * (filter_shape[1] - 1)
        else:
            self.num_padding_zeros = 0

        self.input_shape = None
        self.current_padded_input = None

    def forward_prop(self, input):
        assert self.input_shape == input.shape, "Input does not have correct shape"

        padded_input = np.zeros((self.input_shape[0], self.input_shape[1] + self.num_padding_zeros))
        padded_input[:, self.num_padding_zeros / 2:
                        self.num_padding_zeros / 2 + self.input_shape[1]] = input
        self.current_padded_input = padded_input

        output = np.empty(self.get_output_shape())

        filter_w = self.filter_shape[1]
        range_w = self.get_output_shape()[1]

        for f in range(self.num_filters):
            for w in range(range_w):
                output[f][w] = np.sum(np.multiply(self.filter_weights[f],
                                                  padded_input[:, w:w+filter_w])) + self.biases[f]

        return output

    def back_prop(self, output_grad):
        padded_input_grad = np.zeros((self.input_shape[0],
                                      self.input_shape[1] + self.num_padding_zeros))

        range_w = self.input_shape[1] + self.num_padding_zeros - self.filter_shape[1] + 1
        filter_w = self.filter_shape[1]
        for w in range(range_w):
            for f in range(self.num_filters):
                padded_input_grad[:, w:w + filter_w] += output_grad[f][w] * self.filter_weights[f]
                self.d_filter_weights[f] += output_grad[f][w] *\
                                            self.current_padded_input[:, w:w + filter_w]

        for f in range(self.num_filters):
            self.d_filter_weights[f] += self.weight_decay * self.filter_weights[f]
        # print "\nWeight derivatives for filter " + str(20) + ":\n", self.d_filter_weights[20]

        self.d_biases += np.sum(output_grad, axis=1)
        # print "\nBiases derivatives:\n", self.d_biases

        return padded_input_grad[:, filter_w - 1:range_w]

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
        self.filter_weights -= learning_rate * self.d_filter_weights
        self.biases -= learning_rate * self.d_biases

        # print "\nUPDATED filter weights:\n", self.filter_weights
        # print "\nUPDATED biases:\n", self.biases

        self.d_filter_weights[...] = 0
        self.d_biases[...] = 0

if __name__ == "__main__":
    dummy_input = np.ones((4, 8))
    print "Input:\n", dummy_input

    layer = ConvLayer(num_filters=3, filter_shape=(4, 3), weight_decay=0.1, weight_scale=0.01,
                      padding_mode=True)
    layer.set_input_shape((4, 8))

    print "\n--->> Forward propagation:\n", layer.forward_prop(dummy_input)

    dummy_output_grad = np.ones((3,10)) / 2
    print "\nOutput gradient:\n", dummy_output_grad

    print "\n--->> Backpropagation:\n", layer.back_prop(dummy_output_grad)

    print "\n--->> Params before update:\n", layer.filter_weights, "\n", layer.biases
    layer.update_parameters(0.01)
    print "\n--->> Params after update:\n", layer.filter_weights, "\n", layer.biases
