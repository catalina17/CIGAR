from layer import Layer


class GlobalPoolingLayer(Layer):

    def __init__(self):
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

    def update_parameters(self, learning_rate):
        raise NotImplementedError()
