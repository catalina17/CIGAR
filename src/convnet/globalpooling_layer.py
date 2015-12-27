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
            # Average pooling
            output[f] = np.mean(input[f])
            # Max pooling
            output[self.input_shape[0] + f] = np.max(input[f])
            self.max_activation_indices[f] = np.argmax(input[f])
            # L2-norm pooling
            output[2 * self.input_shape[0] + f] = np.sqrt(np.sum(input[f] ** 2))
        return output

    def back_prop(self, output_grad):
        mean_output_grad = output_grad[0:self.input_shape[0]]
        max_output_grad = output_grad[self.input_shape[0]:2 * self.input_shape[0]]
        l2_output_grad = output_grad[2 * self.input_shape[0]:]

        input_grad = np.empty(self.input_shape)
        range_f = self.input_shape[0]
        for f in range(range_f):
            # Average grad
            input_grad[f] = mean_output_grad[f] / self.input_shape[0]
            # Max grad
            input_grad[f, self.max_activation_indices[f]] += max_output_grad[f]
            # L2-norm grad
            sqr_root = np.sqrt(np.sum(self.current_input[f] ** 2))
            if sqr_root >= 1e-5:
                input_grad[f] += self.current_input[f] / sqr_root * l2_output_grad[f]

        return input_grad

    def set_input_shape(self, shape):
        """

        Parameters
        ----------
        shape : tuple

        """
        self.input_shape = shape
        # print "GlobalPoolingLayer with input shape " + str(shape)

    def get_output_shape(self):
        """

        Returns
        -------
        tuple

        """
        shape = (3 * self.input_shape[0], )
        # print "GlobalPoolingLayer with output shape " + str(shape)
        return shape

if __name__ == '__main__':
    dummy_input = np.ones((2, 4))
    dummy_input.T[2] = 2 * np.ones((2,))
    print "Input:\n", dummy_input

    layer = GlobalPoolingLayer()
    layer.set_input_shape(dummy_input.shape)

    print "Forward propagation:\n", layer.forward_prop(dummy_input)

    dummy_output_grad = np.ones((12,))
    print "Output gradient:\n", dummy_output_grad

    print "Backpropagation:\n", layer.back_prop(dummy_output_grad)
