import numpy as np
import numpy.testing
from unittest import TestCase

from softmax_layer import SoftmaxLayer


class TestSoftmaxLayer(TestCase):

    def setUp(self):
        self.layer = SoftmaxLayer()
        self.layer.set_input_shape((4, ))

    def test_forward_prop_two_equal(self):
        input = np.array([43265, 0, 0, 43265], dtype=np.float64)
        output = self.layer.forward_prop(input)
        expected_output = np.array([0.5, 0, 0, 0.5], dtype=np.float64)
        numpy.testing.assert_array_almost_equal(output, expected_output)

    def test_forward_prop_one_large(self):
        input = np.array([43265, 2, 54, 21], dtype=np.float64)
        output = self.layer.forward_prop(input)
        expected_output = np.array([1, 0, 0, 0], dtype=np.float64)
        numpy.testing.assert_array_almost_equal(output, expected_output)

    def test_back_prop(self):
        out_grad = np.array([3, 3, 3, 3], dtype=np.float64)
        with self.assertRaises(NotImplementedError):
            self.layer.back_prop(out_grad)

    def test_get_output_shape(self):
        self.layer.set_input_shape((5, ))
        self.assertEqual(self.layer.get_output_shape(), (5, ))

if __name__ == '__main__':
    TestSoftmaxLayer.run()
