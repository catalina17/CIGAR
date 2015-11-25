from layer import Layer


class FullyConnectedLayer(Layer):

    def __init__(self, num_nodes):
        """

        Parameters
        ----------
        num_nodes : int

        """
        self.input_shape = None
        self.num_nodes = num_nodes

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
        return (self.num_nodes, )

    def update_parameters(self, learning_rate):
        """

        Parameters
        ----------
        learning_rate : float

        """
        raise NotImplementedError()
