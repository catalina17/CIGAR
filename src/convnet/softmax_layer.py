import numpy as np

from layer import Layer


class SoftmaxLayer(Layer):

    def __init__(self):
        self._num_nodes = None

    def forward_prop(self, input):
        """

        Parameters
        ----------
        input : array of double
            The input for the layer.

        Returns
        -------
        array of double
            The result of the layer processing the input, representing the output of the network.

        """
        input -= np.amax(input)
        exp = np.exp(input)
        return exp / np.sum(exp)

    def back_prop(self, output_grad):
        raise NotImplementedError("Output layer ---> NO back-propagation; use initial_gradient()" +
                                  " instead")

    @staticmethod
    def initial_gradient(predicted_output, true_output):
        """

        Parameters
        ----------
        predicted_output : array of double
            The predicted probability distribution over genres for the example which has just been
            processed by the network.
        true_output : array of double
            The true probability distribution over genres for the same example.

        Returns
        -------
        array of double
            The initial gradient to be used in the back-propagation phase.

        """
        return predicted_output - true_output

    @staticmethod
    def loss(predicted_output, true_output):
        """

        Parameters
        ----------
        predicted_output : array of double
            The predicted probability distribution over genres for the example which has just been
            processed by the network.
        true_output : array of double
            The true probability distribution over genres for the same example.

        Returns
        -------
        double
            The result of the cross-entropy function applied to the given distributions.

        """
        return -np.sum(true_output * np.log(predicted_output / np.sum(predicted_output)))

    def set_input_shape(self, shape):
        """

        Parameters
        ----------
        shape : tuple
            The shape of the inputs which this layer will process.

        """
        assert len(shape) == 1, "Input shape not suitable for output (Softmax) layer"
        self._num_nodes = shape[0]

    def get_output_shape(self):
        """

        Returns
        -------
        tuple
            The output shape of this layer.

        """
        shape = (self._num_nodes,)
        return shape
