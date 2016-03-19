import numpy as np
import numpy.testing
import unittest

from globalpooling_layer import GlobalPoolingLayer
from globalpooling_layer_cuda import GlobalPoolingLayerCUDA


class TestGlobalPoolingLayer(unittest.TestCase):

    def setUp(self):
        self.layer = GlobalPoolingLayer()
        self.layer.set_input_shape((2, 4))
        self.layer_cuda = GlobalPoolingLayerCUDA()
        self.layer_cuda.set_input_shape((2, 4))
        self.input = np.array([[-5, 1, 7, 5],
                               [-50, -10, -70, 50]], dtype=np.float64)

    def test_forward_prop(self):
        expected_output = np.array([2, -20, 7, 50, 10, 100], dtype=np.float64)

        output = self.layer.forward_prop(self.input)
        numpy.testing.assert_array_equal(output, expected_output)

        output = self.layer_cuda.forward_prop(self.input)
        numpy.testing.assert_array_equal(output, expected_output)

    def test_back_prop(self):
        self.layer.forward_prop(self.input)
        self.layer_cuda.forward_prop(self.input)

        out_grad = np.array([0.1, 0.2, 0.3, 0.2, 0.1, 0.2], dtype=np.float64)
        expected_in_grad = np.array([[0.025 - 0.5 * 0.1, 0.025 + 0.1 * 0.1, 0.025 + 0.3 + 0.7 * 0.1,
                                      0.025 + 0.5 * 0.1],
                                     [0.05 - 0.5 * 0.2, 0.05 - 0.1 * 0.2, 0.05 - 0.7 * 0.2,
                                      0.05 + 0.2 + 0.5 * 0.2]],
                                    dtype=np.float64)

        in_grad = self.layer.back_prop(out_grad)
        numpy.testing.assert_array_almost_equal(in_grad, expected_in_grad)
        in_grad = self.layer_cuda.back_prop(out_grad)
        numpy.testing.assert_array_almost_equal(in_grad, expected_in_grad)

    def test_get_output_shape(self):
        self.assertEqual(self.layer.get_output_shape(), (6, ))
        self.assertEqual(self.layer_cuda.get_output_shape(), (6, ))

if __name__ == '__main__':
    TestGlobalPoolingLayer.run()
