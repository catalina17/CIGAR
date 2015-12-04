from layer import Layer
import numpy as np


class GlobalPoolingLayer(Layer):

    def __init__(self):
        self.input_shape = None
        self.current_input = None

    def forward_prop(self, input):
        assert self.input_shape == input.shape, "Input does not have correct shape"
        self.current_input = input

        output = np.empty(self.get_output_shape())
        for f in range(self.input_shape[0]):
            # Mean pooling
            output[f] = np.mean(input[f])
            # Max pooling
            output[self.input_shape[0] + f] = np.max(input[f])
            # L2 pooling
            output[2 * self.input_shape[0] + f] = np.sqrt(sum(input[f] ** 2))
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

if __name__ == '__main__':
    dummy_input = np.ones((4, 4))
    print "Input: "
    print dummy_input

    layer = GlobalPoolingLayer()
    layer.set_input_shape(dummy_input.shape)

    print "Forward propagation:"
    res = layer.forward_prop(dummy_input)
    print res
    print '\n'
