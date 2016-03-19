import numpy as np
import numpy.testing
import unittest

from fullyconnected_layer import FullyConnectedLayer
from fullyconnected_layer_cuda import FullyConnectedLayerCUDA


class TestFullyConnectedLayer(unittest.TestCase):

    def setUp(self):
        self.layer = FullyConnectedLayer(num_nodes=3, weight_scale=1)
        self.layer.set_input_shape((4, ))
        self.layer._weights = np.array([[1, 0.5, -1],
                                        [1, 0.5, 1],
                                        [1, 0.5, -1],
                                        [1, 0.5, 1]], dtype=np.float64)

        self.layer_cuda = FullyConnectedLayerCUDA(num_nodes=3, weight_scale=1)
        self.layer_cuda.set_input_shape((4, ))
        self.layer_cuda._weights = np.array([[1, 0.5, -1],
                                             [1, 0.5, 1],
                                             [1, 0.5, -1],
                                             [1, 0.5, 1]], dtype=np.float64)

    def test_forward_prop(self):
        input = np.array([-3, 14, -5, 6], dtype=np.float64)
        expected_output = np.array([12, 6, 28], dtype=np.float64)

        output = self.layer.forward_prop(input)
        numpy.testing.assert_array_equal(output, expected_output)
        output = self.layer_cuda.forward_prop(input)
        numpy.testing.assert_array_equal(output, expected_output)

    def test_back_prop(self):
        input = np.array([1, 1, 1, 1], dtype=np.float64)
        self.layer.forward_prop(input)
        self.layer_cuda.forward_prop(input)

        out_grad = np.array([1, 1, 1], dtype=np.float64)
        expected_in_grad = np.array([0.5, 2.5, 0.5, 2.5], dtype=np.float64)

        in_grad = self.layer.back_prop(out_grad)
        numpy.testing.assert_array_equal(in_grad, expected_in_grad)
        in_grad = self.layer_cuda.back_prop(out_grad)
        numpy.testing.assert_array_equal(in_grad, expected_in_grad)

    def test_get_output_shape(self):
        self.assertEqual(self.layer.get_output_shape(), (3, ))
        self.assertEqual(self.layer_cuda.get_output_shape(), (3, ))

    def test_update_parameters(self):
        self.layer._weights = np.ones((4, 3), dtype=np.float64)
        self.layer_cuda._weights = np.ones((4, 3), dtype=np.float64)

        input = np.array([1, 1, 1, 1], dtype=np.float64)
        expected_output = np.array([4, 4, 4], dtype=np.float64)

        output = self.layer.forward_prop(input)
        numpy.testing.assert_array_equal(output, expected_output)
        output = self.layer_cuda.forward_prop(input)
        numpy.testing.assert_array_equal(output, expected_output)

        out_grad = np.array([1, 1, 1], dtype=np.float64)
        self.layer.back_prop(out_grad)
        self.layer_cuda.back_prop(out_grad)

        self.layer.update_parameters(0.1)
        self.layer_cuda.update_parameters(0.1)

        expected_output = np.array([3.5, 3.5, 3.5], dtype=np.float64)

        output = self.layer.forward_prop(input)
        numpy.testing.assert_array_equal(output, expected_output)
        output = self.layer_cuda.forward_prop(input)
        numpy.testing.assert_array_equal(output, expected_output)

if __name__ == '__main__':
    TestFullyConnectedLayer.run()
