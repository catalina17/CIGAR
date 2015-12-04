from layer import Layer
import numpy as np


class GlobalPoolingLayer(Layer):

    def __init__(self):
        self.input_shape = None
        self.current_input = None
        self.max_activation_indices = None

    def forward_prop(self, input):
        assert self.input_shape == input.shape, "Input does not have correct shape"
        self.current_input = input

        output = np.empty(self.get_output_shape())
        self.max_activation_indices = np.empty(self.get_output_shape())

        for f in range(self.input_shape[0]):
            # Mean pooling
            output[f] = np.mean(input[f])
            # Max pooling
            output[self.input_shape[0] + f] = np.max(input[f])
            self.max_activation_indices[f] = np.argmax(input[f])
            # L2 pooling
            output[2 * self.input_shape[0] + f] = np.sqrt(sum(input[f] ** 2))
        return output

    def back_prop(self, output_grad):
        mean_output_grad = output_grad[0:self.input_shape[0]]
        max_output_grad = output_grad[self.input_shape[0]:2 * self.input_shape[0]]
        l2_output_grad = output_grad[2 * self.input_shape:]

        input_grad = np.empty(self.input_shape)
        range_f = self.get_output_shape()[0] / 3
        for f in range(range_f):
            mean_input_grad = mean_output_grad[f] / self.input_shape[1]
            max_input_grad = np.zeros(self.input_shape)
            max_input_grad[self.max_activation_indices[f]] = max_output_grad[f]
            l2_input_grad = self.current_input[f] / np.sqrt(sum(self.current_input[f] ** 2)) *\
                            l2_output_grad[f]

            input_grad[f] = mean_input_grad + max_input_grad + l2_input_grad

        return input_grad

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
