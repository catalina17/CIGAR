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
        self.filter_weights = np.random.rand(filter_shape[0], filter_shape[1])

        self.d_weights = np.zeros(filter_shape)
        self.weight_decay = weight_decay

        if padding_mode:
            self.num_padding_zeros = 2 * (filter_shape[1] - 1)
        else:
            self.num_padding_zeros = 0

        self.input_shape = None

    def forward_prop(self, input):
        raise NotImplementedError()

    def back_prop(self, output_grad):
        raise NotImplementedError()

    def set_input_shape(self, shape):
        """

        Parameters
        ----------
        shape : tuple

        """
        self.input_shape = shape

    def get_output_shape(self):
        """

        Returns
        -------
        tuple

        """
        return (self.num_filters,
                self.input_shape[1] + self.num_padding_zeros - self.filter_shape[1] + 1)

    def update_parameters(self, learning_rate):
        raise NotImplementedError()
