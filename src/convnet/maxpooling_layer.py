from layer import Layer


class MaxPoolingLayer(Layer):
    def __init__(self, filter_shape):
        """

        Parameters
        ----------
        filter_shape : tuple

        """
        self.filter_shape = filter_shape
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
        assert self.input_shape[0] % self.filter_shape[0] == 0 and\
            self.input_shape[1] % self.filter_shape[1] == 0,\
            "Input shape is not a multiple of filter shape in MaxPoolingLayer"

        if self.input_shape[0] / self.filter_shape[0] == 1:
            return (self.input_shape[1] / self.filter_shape[1], )
        else:
            return (self.input_shape[0] / self.filter_shape[0],
                    self.input_shape[1] / self.filter_shape[1])

    def update_parameters(self, learning_rate):
        """

        Parameters
        ----------
        learning_rate : float

        """
        raise NotImplementedError()
