import numpy as np
import time

import pycuda.autoinit
import pycuda.driver as driver
from pycuda.compiler import SourceModule

from maxpooling_layer import MaxPoolingLayer


class MaxPoolingLayerCUDA(MaxPoolingLayer):

    def __init__(self, filter_shape):
        """

        Parameters
        ----------
        filter_shape : tuple
            The shape of the pooling filter (area of disjoint regions over which max pooling will
                                             be done).

        """
        mod = SourceModule("""
        __global__ void multiply_them(float *dest, float *a, float *b) {
            const int i = threadIdx.x;
            dest[i] = a[i] * b[i];
        }
        """)
        super(MaxPoolingLayerCUDA, self).__init__(filter_shape)

    def forward_prop(self, input):
        """

        Parameters
        ----------
        input : array of double
            The input for the layer.

        Returns
        -------
        array of double
            The result of the max pooling layer processing the input.

        """
        assert self._input_shape == input.shape, "Input does not have correct shape"
        self._current_input = input

        if (len(self.get_output_shape()) == 1):
            output = np.empty((1, self.get_output_shape()[0])).astype(np.double)
        else:
            output = np.empty(self.get_output_shape()).astype(np.double)

        self._max_activation_indices = np.empty(self.get_output_shape()).astype(np.int32)

        mod = None
        if self._filter_shape[1] == 4:
            # Pooling regions of shape (1, 4)
            mod = SourceModule("""
                #define INPUT_HEIGHT """ + str(self._input_shape[0]) + """
                #define INPUT_WIDTH """ + str(self._input_shape[1] / 4) + """

                __global__ void max_pool(double4 *in, double *dest, int *max_idx_w) {
                    int col = blockIdx.x * blockDim.x + threadIdx.x;
                    int row = blockIdx.y * blockDim.y;

                    if (col >= INPUT_WIDTH || row >= INPUT_HEIGHT)
                        return;

                    int idx = col + row * INPUT_WIDTH;
                    int start = col << 2;

                    double4 val = in[idx];
                    double max_val = val.x;
                    int max_idx = start;

                    if (max_val < val.y) {
                        max_val = val.y;
                        max_idx = start + 1;
                    }

                    if (max_val < val.z) {
                        max_val = val.z;
                        max_idx = start + 2;
                    }

                    if (max_val < val.w) {
                        max_val = val.w;
                        max_idx = start + 3;
                    }

                    dest[idx] = max_val;
                    max_idx_w[idx] = max_idx;
                }
                """)
        elif self._filter_shape[1] == 2:
            # Pooling regions of shape (1, 2)
            mod = SourceModule("""
                #define INPUT_HEIGHT """ + str(self._input_shape[0]) + """
                #define INPUT_WIDTH """ + str(self._input_shape[1] / 2) + """

                __global__ void max_pool(double2 *in, double *dest, int *max_idx_w) {
                    int col = blockIdx.x * blockDim.x + threadIdx.x;
                    int row = blockIdx.y * blockDim.y;

                    if (col >= INPUT_WIDTH || row >= INPUT_HEIGHT)
                        return;

                    int idx = col + row * INPUT_WIDTH;
                    int start = col << 1;

                    double max_val = in[idx].x;
                    max_idx_w[idx] = start;

                    if (max_val < in[idx].y) {
                        max_val = in[idx].y;
                        max_idx_w[idx] = start + 1;
                    }

                    dest[idx] = max_val;
                }
                """)

        max_pool = mod.get_function('max_pool')
        max_pool(driver.In(input.astype(np.double)), driver.Out(output),
                 driver.Out(self._max_activation_indices), block=(self._filter_shape[1], 1, 1),
                 grid=(self.get_output_shape()[1], self.get_output_shape()[0], 1))

        if output.shape[0] == 1:
            return output[0]
        else:
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
        input_grad = np.zeros(self._input_shape).astype(np.double)

        mod = SourceModule("""
            #define INGRAD_WIDTH """ + str(input_grad.shape[1]) + """

            __global__ void max_pool_back(double *out_grad, double *in_grad,
                                          int *max_activation_indices) {
                int col = blockIdx.x * blockDim.x + threadIdx.x;
                int row = blockIdx.y * blockDim.y + threadIdx.y;

                if (col >= (gridDim.x * blockDim.x) || row >= (gridDim.y * blockDim.x))
                    return;

                int out_idx = col + row * (gridDim.x * blockDim.x);
                in_grad[max_activation_indices[out_idx] + row * INGRAD_WIDTH] =
                    out_grad[out_idx];
            }
            """)

        max_pool_back = mod.get_function('max_pool_back')
        max_pool_back(driver.In(output_grad.astype(np.double)), driver.InOut(input_grad),
                      driver.In(self._max_activation_indices),
                      block=(8, 4, 1),
                      grid=(output_grad.shape[1] / 8, output_grad.shape[0] / 4, 1))

        return input_grad

    def set_input_shape(self, shape):
        super(MaxPoolingLayerCUDA, self).set_input_shape(shape)

    def get_output_shape(self):
        return super(MaxPoolingLayerCUDA, self).get_output_shape()
