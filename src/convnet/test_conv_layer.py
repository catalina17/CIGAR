import numpy as np
import numpy.testing
import unittest

from conv_layer import ConvLayer
from conv_layer_cuda import ConvLayerCUDA


class TestConvLayer(unittest.TestCase):

    def setUp(self):
        self.layer = ConvLayer(2, (2, 3), 1, padding_mode=False)
        self.layer.set_input_shape((2, 8))
        self.layer._filter_weights = np.ones((2, 2, 3), dtype=np.double)
        self.layer._filter_weights[1, :, :] /= 2.0

        self.layer_cuda = ConvLayerCUDA(2, (2, 3), 1, padding_mode=False)
        self.layer_cuda.set_input_shape((2, 8))
        self.layer_cuda._filter_weights = np.ones((2, 2, 3), dtype=np.double)
        self.layer_cuda._filter_weights[1, :, :] /= 2.0

        self.input = np.array([[2, 2, 2, 2, 2, 2, 2, 4],
                               [4, 4, 4, 4, 4, 4, 4, 8]], dtype=np.double)

    def test_forward_prop(self):
        expected_output = np.array([[18, 18, 18, 18, 18, 24],
                                    [9, 9, 9, 9, 9, 12]], dtype=np.double)

        output = self.layer.forward_prop(self.input)
        numpy.testing.assert_array_equal(output, expected_output)
        output = self.layer_cuda.forward_prop(self.input)
        numpy.testing.assert_array_equal(output, expected_output)

    def test_back_prop(self):
        self.layer.forward_prop(self.input)
        self.layer_cuda.forward_prop(self.input)

        out_grad = np.ones((2, 6), dtype=np.double)
        expected_in_grad = np.array([[],
                                     []], dtype=np.double)

    def test_get_output_shape(self):
        self.assertEqual(self.layer.get_output_shape(), (2, 6))
        self.assertEqual(self.layer_cuda.get_output_shape(), (2, 6))

    def test_update_parameters(self):
        self.fail()

if __name__ == '__main__':
    TestConvLayer.run()
