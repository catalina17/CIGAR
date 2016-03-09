from layer import Layer
import numpy as np


class SoftmaxLayer(Layer):

    def __init__(self):
        self._num_nodes = None

    def forward_prop(self, input):
        input -= np.amax(input)
        exp = np.exp(input)
        return exp / np.sum(exp)

    def back_prop(self, output_grad):
        raise NotImplementedError("Output layer -- no backpropagation")

    def initial_gradient(self, predicted_output, true_output):
        return predicted_output - true_output

    def loss(self, predicted_output, true_output):
        return -np.sum(true_output * np.log(predicted_output / np.sum(predicted_output)))

    def set_input_shape(self, shape):
        """

        Parameters
        ----------
        shape : tuple

        """
        assert len(shape) == 1, "Input shape not suitable for output (Softmax) layer"
        self._num_nodes = shape[0]
        # print "SoftmaxLayer with input shape " + str(shape)

    def get_output_shape(self):
        """

        Returns
        -------
        tuple

        """
        shape = (self._num_nodes,)
        # print "SoftmaxLayer with output shape " + str(shape)
        return shape

if __name__ == "__main__":
    dummy_input = np.zeros(3)
    dummy_input[0] = 0
    dummy_input[1] = 1
    dummy_input[2] = 2
    print "Input:\n", dummy_input

    layer = SoftmaxLayer()
    layer.set_input_shape((3,))

    print "\n--->> Forward propagation:\n", layer.forward_prop(dummy_input)

    dummy_output = np.abs(np.random.uniform(0, 1, 3))
    one_hot_outputs = np.array([1, 0, 0])
    print "\n--->> Output:\n", dummy_output
    print "\nGradient:\n", layer.initial_gradient(one_hot_outputs, dummy_output)
    print "\nLoss:\n", layer.loss(dummy_output, one_hot_outputs)
