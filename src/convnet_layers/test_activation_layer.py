import numpy as np
import numpy.testing
import unittest

from activation_layer import ActivationLayer
from activation_layer_cuda import ActivationLayerCUDA


class TestActivationLayer(unittest.TestCase):

    def test_forward_prop_relu(self):
        layer = ActivationLayer('ReLU')
        layer.set_input_shape((4, ))

        input = np.array([1536, -83741, 84597, -1543], dtype=np.float64)
        expected_output = np.array([1536, 0, 84597, 0], dtype=np.float64)

        output = layer.forward_prop(input)
        numpy.testing.assert_array_equal(output, expected_output)

    def test_forward_prop_leaky_relu(self):
        layer = ActivationLayer('leakyReLU')
        layer.set_input_shape((4, ))

        input = np.array([57287, -6154385, 14938, -54787], dtype=np.float64)
        expected_output = np.array([57287, -61543.85, 14938, -547.87], dtype=np.float64)

        output = layer.forward_prop(input)
        numpy.testing.assert_array_almost_equal(output, expected_output)

    def test_forward_prop_sigmoid(self):
        layer = ActivationLayer('sigmoid')
        layer.set_input_shape((4, ))

        input = np.array([0, 0, 0, 0], dtype=np.float64)
        expected_output = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float64)

        output = layer.forward_prop(input)
        numpy.testing.assert_array_equal(output, expected_output)

    def test_forward_prop_sigmoid_large_input(self):
        layer = ActivationLayer('sigmoid')
        layer.set_input_shape((4, ))

        input = np.array([427562359, 4329756, 7686586, 0], dtype=np.float64)
        expected_output = np.array([1, 1, 1, 0.5], dtype=np.float64)

        output = layer.forward_prop(input)
        numpy.testing.assert_array_almost_equal(output, expected_output)

    def test_forward_prop_sigmoid_small_input(self):
        layer = ActivationLayer('sigmoid')
        layer.set_input_shape((4, ))

        input = np.array([-427562359, -329756, -86, 0], dtype=np.float64)
        expected_output = np.array([0, 0, 0, 0.5], dtype=np.float64)

        output = layer.forward_prop(input)
        numpy.testing.assert_array_almost_equal(output, expected_output)

    def test_back_prop_relu(self):
        layer = ActivationLayer('ReLU')
        layer.set_input_shape((4, ))

        input = np.array([1, -1, 1, -1], dtype=np.float64)
        layer.forward_prop(input)

        expected_in_grad = np.array([1, 0, 1, 0], dtype=np.float64)
        out_grad = np.ones(4)

        in_grad = layer.back_prop(out_grad)
        numpy.testing.assert_array_almost_equal(in_grad, expected_in_grad)

    def test_back_prop_leaky_relu(self):
        layer = ActivationLayer('leakyReLU')
        layer.set_input_shape((4, ))

        input = np.array([1, -1, 1, -1], dtype=np.float64)
        layer.forward_prop(input)

        out_grad = np.ones(4)
        expected_in_grad = np.array([1, 0.01, 1, 0.01], dtype=np.float64)

        in_grad = layer.back_prop(out_grad)
        numpy.testing.assert_array_almost_equal(in_grad, expected_in_grad)

    def test_back_prop_sigmoid(self):
        layer = ActivationLayer('sigmoid')
        layer.set_input_shape((4, ))

        input = np.array([0, 2365836, 0, -154366], dtype=np.float64)
        layer.forward_prop(input)

        expected_in_grad = np.array([0.25, 0, 0.25, 0], dtype=np.float64)
        out_grad = np.ones(4)

        in_grad = layer.back_prop(out_grad)
        numpy.testing.assert_array_almost_equal(in_grad, expected_in_grad)

    def test_get_output_shape(self):
        layer = ActivationLayer('ReLU')
        layer.set_input_shape((100, ))

        self.assertEqual(layer.get_output_shape(), (100, ))

if __name__ == '__main__':
    TestActivationLayer.run()
