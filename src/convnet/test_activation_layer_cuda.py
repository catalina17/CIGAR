import numpy as np
import numpy.testing
import unittest

from activation_layer_cuda import ActivationLayerCUDA


class TestActivationLayerCUDA(unittest.TestCase):

    def test_forward_prop_leaky_relu(self):
        layer_cuda = ActivationLayerCUDA('leakyReLU')
        layer_cuda.set_input_shape((4, ))

        input = np.array([57287, -6154385, 14938, -54787], dtype=np.float64)
        expected_output = np.array([57287, -61543.85, 14938, -547.87], dtype=np.float64)

        output = layer_cuda.forward_prop(input)
        numpy.testing.assert_array_almost_equal(output, expected_output)

    def test_back_prop_leaky_relu(self):
        layer_cuda = ActivationLayerCUDA('leakyReLU')
        layer_cuda.set_input_shape((4, ))

        input = np.array([1, -1, 1, -1], dtype=np.float64)
        layer_cuda.forward_prop(input)

        out_grad = np.ones(4)
        expected_in_grad = np.array([1, 0.01, 1, 0.01], dtype=np.float64)

        in_grad = layer_cuda.back_prop(out_grad)
        numpy.testing.assert_array_almost_equal(in_grad, expected_in_grad)

    def test_get_output_shape(self):
        layer_cuda = ActivationLayerCUDA('ReLU')
        layer_cuda.set_input_shape((100, ))

        self.assertEqual(layer_cuda.get_output_shape(), (100, ))

if __name__ == '__main__':
    TestActivationLayerCUDA.run()
