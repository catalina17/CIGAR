import numpy as np
import time

import pycuda.autoinit
import pycuda.driver as driver
from pycuda.compiler import SourceModule

from conv_layer import ConvLayer


class ConvLayerCUDA(ConvLayer):

    def __init__(self, num_filters, filter_shape, weight_scale, padding_mode=True):
        """

        Parameters
        ----------
        num_filters : int
            The number of filters for this layer.
        filter_shape : tuple
            The shape of the filters.
        weight_scale : float
            The standard deviation sigma of the normal distribution N(0, sigma^2) from which
            the filter weights are initialised.
        padding_mode : bool
            Whether the input is padded with columns of zeros.

        """
        mod = SourceModule("""
            __global__ void multiply_them(float *dest, float *a, float *b) {
                const int i = threadIdx.x;
                dest[i] = a[i] * b[i];
            }
            """)
        super(ConvLayerCUDA, self).__init__(num_filters, filter_shape, weight_scale, padding_mode)

    def forward_prop(self, input):
        """

        Parameters
        ----------
        input : array of double
            The input for the layer.

        Returns
        -------
        array of double
            The result of the convolutional layer processing the input.

        """
        assert self._input_shape == input.shape, "Input does not have correct shape"

        padded_input = np.zeros((self._input_shape[0],
                                 self._input_shape[1] + self._num_padding_zeros)).astype(np.double)
        padded_input[:, self._num_padding_zeros / 2:
                        self._num_padding_zeros / 2 + self._input_shape[1]] = input
        self._current_padded_input = padded_input

        output = np.empty(self.get_output_shape()).astype(np.double)

        mod = SourceModule("""
            #define INPUT_WIDTH """ + str(self._input_shape[1]) + """
            #define FILTER_AREA """ + str(self._filter_shape[0] * self._filter_shape[1]) + """

            __global__ void conv_fwd_prop(double *in, double *f_weights, double *biases,
                                          double *out) {
                __shared__ double temp[FILTER_AREA];

                temp[blockDim.x * threadIdx.y + threadIdx.x] =
                    in[threadIdx.y * INPUT_WIDTH + blockIdx.x + threadIdx.x] *
                        f_weights[blockIdx.y * blockDim.x * blockDim.y + threadIdx.y * blockDim.x +
                                  threadIdx.x];

                __syncthreads();

                double sum = 0.0;
                for (int i = 0; i < FILTER_AREA; ++i)
                    sum += temp[i];
                out[blockIdx.y * gridDim.x + blockIdx.x] = sum + biases[blockIdx.y];
            }
            """)
        conv_fwd_prop = mod.get_function('conv_fwd_prop')
        conv_fwd_prop(driver.In(padded_input), driver.In(self._filter_weights),
                      driver.In(self._biases), driver.InOut(output),
                      block=(self._filter_shape[1], self._filter_shape[0], 1),
                      grid=(self.get_output_shape()[1], self._num_filters, 1))

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
        padded_input_grad = np.zeros((self._input_shape[0],
                                      self._input_shape[1] + self._num_padding_zeros)).\
            astype(np.double)

        mod1 = SourceModule("""
            #define FILTER_HEIGHT """ + str(self._filter_shape[0]) + """
            #define FILTER_WIDTH """ + str(self._filter_shape[1]) + """
            #define NUM_FILTERS """ + str(self._num_filters) + """
            #define OUT_WIDTH """ + str(self.get_output_shape()[1]) + """

            __global__ void compute_in_grad(double *out_grad, double *in,
                                double *f_weights, double *in_grad) {
                __shared__ double temp[FILTER_WIDTH * NUM_FILTERS];

                // Check that the filter completely lies inside the grid
                // given by the input shape of the layer.
                if (blockIdx.x - threadIdx.x < 0 ||
                    blockIdx.x - threadIdx.x >= OUT_WIDTH) {
                    return;
                }

                // Compute the locations of the current thread in the incoming
                // gradient and in the filter weight matrix.
                int out_idx = OUT_WIDTH * threadIdx.y + blockIdx.x - threadIdx.x;
                int f_weights_idx = threadIdx.y * FILTER_HEIGHT * FILTER_WIDTH + threadIdx.x;
                // Compute the current contribution to the gradient.
                int idx = threadIdx.y * blockDim.x + threadIdx.x;
                temp[idx] = out_grad[out_idx] * f_weights[f_weights_idx];

                __syncthreads();

                double sum = 0.0;
                int i_max = FILTER_WIDTH * NUM_FILTERS;
                // Sum over the shared memory in the block to compute the value
                // at the corresponding location in the new gradient.
                for (int i = 0; i < i_max; ++i) {
                    sum += temp[i];
                }
                in_grad[blockIdx.y * gridDim.x + blockIdx.x] = sum;
            }
            """)
        compute_in_grad = mod1.get_function('compute_in_grad')
        compute_in_grad(driver.In(output_grad.astype(np.double)),
                        driver.In(self._current_padded_input),
                        driver.In(self._filter_weights),
                        driver.InOut(padded_input_grad),
                        block=(self._filter_shape[1], self._num_filters, 1),
                        grid=(padded_input_grad.shape[1], padded_input_grad.shape[0], 1))

        mod2 = SourceModule("""
            #define OUT_WIDTH """ + str(self.get_output_shape()[1]) + """
            #define IN_WIDTH """ + str(self._input_shape[1]) + """
            #define FILTER_HEIGHT """ + str(self._filter_shape[0]) + """
            #define FILTER_WIDTH """ + str(self._filter_shape[1]) + """

            __global__ void compute_derivatives(double *out_grad, double *in, double *d_f_weights,
                                                double *d_biases) {
                int idx = FILTER_WIDTH * FILTER_HEIGHT * blockIdx.x + FILTER_WIDTH * threadIdx.y +
                          threadIdx.x;
                int in_idx = IN_WIDTH * threadIdx.y + threadIdx.x;
                int out_idx = OUT_WIDTH * blockIdx.x;
                double d_bias = 0.0;

                for (int i = 0; i < OUT_WIDTH; ++i) {
                    double val = out_grad[out_idx + i];
                    d_f_weights[idx] += val * in[in_idx + i];

                    if (!threadIdx.x && !threadIdx.y) {
                        d_bias += val;
                    }
                }

                if (!threadIdx.x && !threadIdx.y) {
                    d_biases[blockIdx.x] += d_bias;
                }
            }
            """)
        compute_derivatives = mod2.get_function('compute_derivatives')
        compute_derivatives(driver.In(output_grad.astype(np.double)),
                            driver.In(self._current_padded_input),
                            driver.InOut(self._d_filter_weights), driver.InOut(self._d_biases),
                            block=(self._filter_shape[1], self._filter_shape[0], 1),
                            grid=(self._num_filters, 1, 1))

        return padded_input_grad[:, self._num_padding_zeros / 2:self._input_shape[1] +
                                                                self._num_padding_zeros / 2]

    def set_input_shape(self, shape):
        super(ConvLayerCUDA, self).set_input_shape(shape)

    def get_output_shape(self):
        return super(ConvLayerCUDA, self).get_output_shape()

    def update_parameters(self, learning_rate):
        super(ConvLayerCUDA, self).update_parameters(learning_rate)

    def serialise_parameters(self, file_idx):
        super(ConvLayerCUDA, self).serialise_parameters(file_idx)

    def init_parameters_from_file(self, file_idx):
        super(ConvLayerCUDA, self).init_parameters_from_file(file_idx)
