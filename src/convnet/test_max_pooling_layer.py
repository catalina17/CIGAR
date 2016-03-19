import numpy as np
import numpy.testing
import unittest

from maxpooling_layer import MaxPoolingLayer
from maxpooling_layer_cuda import MaxPoolingLayerCUDA


class TestMaxPoolingLayer(unittest.TestCase):

    def setUp(self):
        self.layer = MaxPoolingLayer((1, 2))
        self.layer_cuda = MaxPoolingLayerCUDA((1, 2))

    def test_forward_prop(self):
        self.layer.set_input_shape((4, 4))
        self.layer_cuda.set_input_shape((4, 4))

        input = np.array([[-1, -2, 3, 4],
                          [5, 6, -7, -8],
                          [9, -10, 11, -12],
                          [-13, 14, -15, 16]], dtype=np.float64)
        expected_output = np.array([[-1, 4],
                                    [6, -7],
                                    [9, 11],
                                    [14, 16]], dtype=np.float64)

        output = self.layer.forward_prop(input)
        numpy.testing.assert_array_equal(output, expected_output)
        output = self.layer_cuda.forward_prop(input)
        numpy.testing.assert_array_equal(output, expected_output)

    def test_back_prop(self):
        self.layer.set_input_shape((4, 16))
        self.layer_cuda.set_input_shape((4, 16))

        input = np.array([[1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
                          [3, 4, -3, -4, 3, 4, -3, -4, 3, 4, -3, -4, 3, 4, -3, -4],
                          [-3, -4, 3, 4, -3, -4, 3, 4, -3, -4, 3, 4, -3, -4, 3, 4],
                          [1, 2, -1, -2, 1, 2, -1, -2, 1, 2, -1, -2, 1, 2, -1, -2]],
                         dtype=np.float64)

        self.layer.forward_prop(input)
        self.layer_cuda.forward_prop(input)

        out_grad = np.array([[1, 1, 1, 1, 1, 1, 1, 1],
                             [1, 1, 1, 1, 1, 1, 1, 1],
                             [1, 1, 1, 1, 1, 1, 1, 1],
                             [1, 1, 1, 1, 1, 1, 1, 1]])
        expected_in_grad = np.array([[0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                                     [0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0],
                                     [1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1],
                                     [0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0]],
                                    dtype=np.float64)

        in_grad = self.layer.back_prop(out_grad)
        numpy.testing.assert_array_equal(in_grad, expected_in_grad)
        in_grad = self.layer_cuda.back_prop(out_grad)
        numpy.testing.assert_array_equal(in_grad, expected_in_grad)

    def test_get_output_shape(self):
        self.layer.set_input_shape((280, 72))
        self.layer_cuda.set_input_shape((280, 72))

        self.assertEqual(self.layer.get_output_shape(), (280, 36))
        self.assertEqual(self.layer_cuda.get_output_shape(), (280, 36))

if __name__ == '__main__':
    TestMaxPoolingLayer.run()
