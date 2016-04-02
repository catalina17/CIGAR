import numpy as np
import numpy.testing
import unittest

from conv_layer_cuda import ConvLayerCUDA


class TestConvLayerCUDA(unittest.TestCase):

    def setUp(self):
        self.layer_cuda = ConvLayerCUDA(2, (2, 3), 1, padding_mode=False)
        self.layer_cuda.set_input_shape((2, 8))
        self.layer_cuda._filter_weights = np.ones((2, 2, 3), dtype=np.double)
        self.layer_cuda._filter_weights[1, :, :] /= 2.0

        self.input = np.array([[2, 2, 2, 2, 2, 2, 2, 4],
                               [4, 4, 4, 4, 4, 4, 4, 8]], dtype=np.double)

    def test_forward_prop(self):
        expected_output = np.array([[18, 18, 18, 18, 18, 24],
                                    [9, 9, 9, 9, 9, 12]], dtype=np.double)

        output = self.layer_cuda.forward_prop(self.input)
        numpy.testing.assert_array_equal(output, expected_output)

    def test_back_prop(self):
        self.layer_cuda.forward_prop(self.input)

        out_grad = np.ones((2, 6), dtype=np.double)
        expected_in_grad = np.array([[1.5, 3, 4.5, 4.5, 4.5, 4.5, 3, 1.5],
                                     [1.5, 3, 4.5, 4.5, 4.5, 4.5, 3, 1.5]], dtype=np.double)

        in_grad = self.layer_cuda.back_prop(out_grad)
        numpy.testing.assert_array_equal(in_grad, expected_in_grad)

    def test_get_output_shape(self):
        self.assertEqual(self.layer_cuda.get_output_shape(), (2, 6))

    def test_update_parameters(self):
        self.layer_cuda.forward_prop(self.input)

        out_grad = np.ones((2, 6), dtype=np.double)

        self.layer_cuda.back_prop(out_grad)

        self.layer_cuda.update_parameters(0.1)

        expected_output = np.array([[-20.6, -20.6, -20.6, -20.6, -20.6, -28.6],
                                    [-29.6, -29.6, -29.6, -29.6, -29.6, -40.6]], dtype=np.double)

        output = self.layer_cuda.forward_prop(self.input)
        numpy.testing.assert_array_almost_equal(output, expected_output)

if __name__ == '__main__':
    TestConvLayerCUDA.run()
