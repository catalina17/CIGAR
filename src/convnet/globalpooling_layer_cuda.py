import numpy as np
import time

import pycuda.autoinit
import pycuda.driver as driver
from pycuda.compiler import SourceModule

from globalpooling_layer import GlobalPoolingLayer


class GlobalPoolingLayerCUDA(GlobalPoolingLayer):

    def __init__(self):
        mod = SourceModule("""
            __global__ void multiply_them(float *dest, float *a, float *b) {
                const int i = threadIdx.x;
                dest[i] = a[i] * b[i];
            }
            """)
        self._l2_values = None
        super(GlobalPoolingLayerCUDA, self).__init__()

    def forward_prop(self, input):
        """

        Parameters
        ----------
        input : array of double
            The input for the layer.

        Returns
        -------
        array of double
            The result of the global pooling layer processing the input.

        """
        assert self._input_shape == input.shape, "Input does not have correct shape"
        self._current_input = input

        output = np.empty(self.get_output_shape()).astype(np.float64)
        # self._max_activation_indices = np.empty(self._input_shape).astype(np.int32)
        self._l2_values = np.empty(self._input_shape[0]).astype(np.float64)

        mod = SourceModule("""
            #define INPUT_LENGTH """ + str(self._input_shape[1]) + """

            __global__ void global_pool(double *in, double *out, int *max_idx, double *l2) {
                __shared__ double temp[INPUT_LENGTH];
                temp[threadIdx.x] = in[blockIdx.x * blockDim.x + threadIdx.x];

                __syncthreads();
                if (threadIdx.x)
                    return;

                int i_max = 0;
                double max_val = temp[0];
                double sum = 0.0;
                double sum_sqr = 0.0;

                for (int i = 0; i < blockDim.x; ++i) {
                    sum += temp[i];
                    sum_sqr += temp[i] * temp[i];
                    if (max_val < temp[i]) {
                        max_val = temp[i];
                        i_max = i;
                    }
                }

                out[blockIdx.x] = sum / (double)blockDim.x;

                out[blockIdx.x + gridDim.x] = max_val;
                max_idx[blockIdx.x] = i_max;

                l2[blockIdx.x] = sqrt(sum_sqr);
                out[blockIdx.x + (gridDim.x << 1)] = l2[blockIdx.x];
            }
            """)
        global_pool = mod.get_function('global_pool')
        global_pool(driver.In(input.astype(np.float64)),
                    driver.InOut(output), driver.InOut(self._max_activation_indices),
                    driver.InOut(self._l2_values),
                    block=(self._input_shape[1], 1, 1), grid=(self._input_shape[0], 1, 1))

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
        input_grad = np.empty(self._input_shape).astype(np.float64)

        mod = SourceModule("""
            __global__ void back_prop(double *out_grad, double *in, double *l2, int *max_idx,
                                      double *in_grad) {
                in_grad[blockIdx.x * blockDim.x + threadIdx.x] = out_grad[blockIdx.x] /
                    (double)blockDim.x;

                if (!threadIdx.x)
                    in_grad[blockIdx.x * blockDim.x + max_idx[blockIdx.x]] +=
                        out_grad[blockIdx.x + gridDim.x];

                if (l2[blockIdx.x] >= 1e-5)
                    in_grad[blockIdx.x * blockDim.x + threadIdx.x] +=
                        in[blockIdx.x * blockDim.x + threadIdx.x] / (double)l2[blockIdx.x] *
                            out_grad[blockIdx.x + (gridDim.x << 1)];
            }
            """)
        back_prop = mod.get_function('back_prop')
        back_prop(driver.In(output_grad.astype(np.float64)),
                  driver.In(self._current_input.astype(np.float64)), driver.In(self._l2_values),
                  driver.In(self._max_activation_indices), driver.InOut(input_grad),
                  block=(self._input_shape[1], 1, 1), grid=(self._input_shape[0], 1, 1))

        return input_grad

    def set_input_shape(self, shape):
        super(GlobalPoolingLayerCUDA, self).set_input_shape(shape)

    def get_output_shape(self):
        return super(GlobalPoolingLayerCUDA, self).get_output_shape()
