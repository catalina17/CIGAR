import numpy as np

import pycuda.driver as driver
import pycuda.autoinit

from activation_layer import ActivationLayer
from pycuda.compiler import SourceModule


class ActivationLayerCUDA(ActivationLayer):

    def __init__(self, activation_fn):
        """

        Parameters
        ----------
        activation_fn : str
            The name of the activation function for this layer.

        """
        mod = SourceModule("""
            __global__ void multiply_them(float *dest, float *a, float *b) {
                const int i = threadIdx.x;
                dest[i] = a[i] * b[i];
            }
            """)
        super(ActivationLayerCUDA, self).__init__(activation_fn)

    def forward_prop(self, input):
        """

        Parameters
        ----------
        input : array of double
            The input for the layer.

        Returns
        -------
        array of double
            The result of the layer applying the activation function to the input.

        """
        assert self._input_shape == input.shape, "Input does not have correct shape"
        self._current_input = input

        if len(self._input_shape) == 1:
            h = 1
            w = self._input_shape[0]
        else:
            h, w = self._input_shape

        mod = SourceModule("""
            #define INPUT_HEIGHT """ + str(h) + """
            #define INPUT_WIDTH """ + str(w) + """

            __global__ void fwd_leakyrelu(double *in, double *out) {
                int col = blockIdx.x * blockDim.x + threadIdx.x;
                int row = blockIdx.y * blockDim.y + threadIdx.y;

                if (col >= INPUT_WIDTH || row >= INPUT_HEIGHT)
                    return;

                int idx = col + row * INPUT_WIDTH;
                out[idx] = in[idx];
                if (in[idx] <= 0.0) {
                    out[idx] *= 0.01;
                }
            }
            """)

        fwd_leakyrelu = mod.get_function('fwd_leakyrelu')
        output = np.empty(self._input_shape).astype(np.float64)
        fwd_leakyrelu(driver.In(input.astype(np.float64)),
                      driver.Out(output),
                      block=(8, 4, 1), grid=(w / 8 + 1, h / 4 + 1, 1))

        return output

    def back_prop(self, output_grad):
        """

        Parameters
        ----------
        output_grad : array of double
            The incoming gradient from the next layer of the network.

        Returns
        -------
        array of double
            The gradient computed by this layer.

        """
        if len(self._input_shape) == 1:
            h = 1
            w = self._input_shape[0]
        else:
            h, w = self._input_shape

        mod = SourceModule("""
            #define INPUT_HEIGHT """ + str(h) + """
            #define INPUT_WIDTH """ + str(w) + """

            __global__ void backprop_leakyrelu(double *grad_out, double *grad_in, double *in) {
                int col = blockIdx.x * blockDim.x + threadIdx.x;
                int row = blockIdx.y * blockDim.y + threadIdx.y;

                if (col >= INPUT_WIDTH || row >= INPUT_HEIGHT)
                    return;

                int idx = col + row * INPUT_WIDTH;
                grad_in[idx] = grad_out[idx];
                if (in[idx] <= 0.0)
                    grad_in[idx] *= 0.01;
            }
            """)

        backprop_leakyrelu = mod.get_function('backprop_leakyrelu')
        input_grad = np.empty(self._input_shape).astype(np.float64)
        backprop_leakyrelu(driver.In(output_grad.astype(np.float64)),
                           driver.Out(input_grad),
                           driver.In(self._current_input.astype(np.float64)),
                           block=(8, 4, 1), grid=(w / 8 + 1, h / 4 + 1, 1))

        return input_grad

    def set_input_shape(self, shape):
        super(ActivationLayerCUDA, self).set_input_shape(shape)

    def get_output_shape(self):
        return super(ActivationLayerCUDA, self).get_output_shape()
