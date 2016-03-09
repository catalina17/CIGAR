import numpy as np
import time

from layer import Layer


class ConvLayer(Layer):

    def __init__(self, num_filters, filter_shape, weight_scale, padding_mode=True):
        """

        Parameters
        ----------
        num_filters : int
        filter_shape : tuple
        weight_scale : float
        padding_mode : bool

        """
        self._num_filters = num_filters
        self._filter_shape = filter_shape
        print "Filter shape: " + str(filter_shape)

        self._filter_weights = np.empty((num_filters, filter_shape[0], filter_shape[1])).\
            astype(np.double)
        for i in range(num_filters):
            self._filter_weights[i] = np.random.normal(loc=0, scale=weight_scale,
                                                       size=filter_shape).astype(np.double)
        self._d_filter_weights = np.zeros(self._filter_weights.shape).astype(np.double)

        self._biases = np.zeros(num_filters).astype(np.double)
        self._d_biases = np.zeros(num_filters).astype(np.double)

        if padding_mode:
            # Padding input with columns
            self._num_padding_zeros = 2 * (filter_shape[1] - 1)
        else:
            self._num_padding_zeros = 0

        self._input_shape = None
        self._current_padded_input = None

    def forward_prop(self, input):
        assert self._input_shape == input.shape, "Input does not have correct shape"

        padded_input = np.zeros((self._input_shape[0],
                                 self._input_shape[1] + self._num_padding_zeros))
        padded_input[:, self._num_padding_zeros / 2:
                        self._num_padding_zeros / 2 + self._input_shape[1]] = input
        self._current_padded_input = padded_input

        output = np.empty(self.get_output_shape())

        filter_w = self._filter_shape[1]
        range_w = self.get_output_shape()[1]

        for f in range(self._num_filters):
            for w in range(range_w):
                output[f][w] = np.sum(np.multiply(self._filter_weights[f],
                                                  padded_input[:, w:w+filter_w])) + self._biases[f]

        return output

    def back_prop(self, output_grad):
        padded_input_grad = np.zeros((self._input_shape[0],
                                      self._input_shape[1] + self._num_padding_zeros))

        range_w = self.get_output_shape()[1]
        filter_w = self._filter_shape[1]
        for w in range(range_w):
            for f in range(self._num_filters):
                padded_input_grad[:, w:w + filter_w] += output_grad[f][w] * self._filter_weights[f]
                self._d_filter_weights[f] += output_grad[f][w] * \
                                             self._current_padded_input[:, w:w + filter_w]

        self._d_biases += np.sum(output_grad, axis=1)

        return padded_input_grad[:, self._num_padding_zeros / 2:self._input_shape[1] +
                                                                self._num_padding_zeros / 2]

    def set_input_shape(self, shape):
        """

        Parameters
        ----------
        shape : tuple

        """
        self._input_shape = shape
        # print "ConvLayer with input shape " + str(shape)

    def get_output_shape(self):
        """

        Returns
        -------
        tuple

        """
        shape = (self._num_filters,
                 self._input_shape[1] + self._num_padding_zeros - self._filter_shape[1] + 1)
        # print "ConvLayer with output shape " + str(shape)
        return shape

    def update_parameters(self, learning_rate):
        self._filter_weights -= learning_rate * self._d_filter_weights
        self._biases -= learning_rate * self._d_biases

        self._d_filter_weights[...] = 0
        self._d_biases[...] = 0

    def serialise_parameters(self, file_idx):
        param_file = open('saved_params/Conv_' + str(file_idx) + '_weights', 'w')
        np.save(param_file, self._filter_weights)
        param_file.close()

        param_file = open('saved_params/Conv_' + str(file_idx) + '_biases', 'w')
        np.save(param_file, self._biases)
        param_file.close()

    def init_parameters_from_file(self, file_idx):
        param_file = open('saved_params/Conv_' + str(file_idx) + '_weights', 'rb')
        self._filter_weights = np.load(param_file)
        param_file.close()

        param_file = open('saved_params/Conv_' + str(file_idx) + '_biases', 'rb')
        self._biases = np.load(param_file)
        param_file.close()

if __name__ == "__main__":
    dummy_input = np.ones((128, 599))
    # print "Input:\n", dummy_input

    layer = ConvLayer(num_filters=32, filter_shape=(128, 4), weight_scale=0.01, padding_mode=False)
    layer.set_input_shape((128, 599))

    start = time.time()
    print "\n--->> Forward propagation:\n", layer.forward_prop(dummy_input)
    finish = time.time()
    print "Fwd prop - time taken: ", finish - start

    dummy_output_grad = np.ones((32, 596)) / 2
    # print "\nOutput gradient:\n", dummy_output_grad

    start = time.time()
    print "\n--->> Backpropagation:\n", layer.back_prop(dummy_output_grad)
    finish = time.time()
    print "Back prop - time taken: ", finish - start
