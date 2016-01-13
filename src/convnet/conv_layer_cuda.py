from conv_layer import ConvLayer
from pycuda.compiler import SourceModule

import pycuda.driver as driver
import pycuda.autoinit
import numpy as np
import time


class ConvLayerCUDA(ConvLayer):

    def __init__(self, num_filters, filter_shape, weight_decay, weight_scale, padding_mode=True):
        mod = SourceModule("""
            __global__ void multiply_them(float *dest, float *a, float *b) {
                const int i = threadIdx.x;
                dest[i] = a[i] * b[i];
            }
            """)
        super(ConvLayerCUDA, self).__init__(num_filters, filter_shape, weight_decay, weight_scale,
                                            padding_mode)

    def forward_prop(self, input):
        assert self.input_shape == input.shape, "Input does not have correct shape"

        padded_input = np.zeros((self.input_shape[0],
                                 self.input_shape[1] + self.num_padding_zeros)).astype(np.float32)
        padded_input[:, self.num_padding_zeros / 2:
                        self.num_padding_zeros / 2 + self.input_shape[1]] = input
        self.current_padded_input = padded_input

        output = np.empty(self.get_output_shape()).astype(np.float32)

        mod = SourceModule("""
            #define INPUT_WIDTH """ + str(self.input_shape[1]) + """
            #define FILTER_AREA """ + str(self.filter_shape[0] * self.filter_shape[1]) + """

            __global__ void conv_fwd_prop(float *in, float *f_weights, float *biases, float *out) {
                __shared__ float temp[FILTER_AREA];

                temp[blockDim.x * threadIdx.y + threadIdx.x] =
                    in[threadIdx.y * INPUT_WIDTH + blockIdx.x + threadIdx.x] *
                        f_weights[blockIdx.y * blockDim.x * blockDim.y + threadIdx.y * blockDim.x +
                                  threadIdx.x];

                __syncthreads();

                float sum = 0.0;
                for (int i = 0; i < FILTER_AREA; ++i)
                    sum += temp[i];
                out[blockIdx.y * gridDim.x + blockIdx.x] = sum + biases[blockIdx.y];
            }
            """)
        conv_fwd_prop = mod.get_function('conv_fwd_prop')
        conv_fwd_prop(driver.In(padded_input), driver.In(self.filter_weights),
                      driver.In(self.biases), driver.InOut(output),
                      block=(self.filter_shape[1], self.filter_shape[0], 1),
                      grid=(self.get_output_shape()[1], self.num_filters, 1))

        return output

    def back_prop(self, output_grad):
        padded_input_grad = np.zeros((self.input_shape[0],
                                      self.input_shape[1] + self.num_padding_zeros)).\
            astype(np.float32)

        range_w = self.get_output_shape()[1]
        filter_w = self.filter_shape[1]

        mod = SourceModule("""
            """)
        conv_back_prop = mod.get_function('conv_back_prop')
        conv_back_prop(driver.In(output_grad.astype(np.float32)),
                       driver.In(self.current_padded_input),
                       driver.In(self.filter_weights),
                       driver.InOut(padded_input_grad),
                       driver.InOut(self.d_filter_weights), driver.InOut(self.d_biases),
                       block=(self.filter_shape[1], self.num_filters, 1),
                       grid=(padded_input_grad.shape[1], padded_input_grad.shape[0], 1))

        return padded_input_grad[:, filter_w - 1:range_w]

    def set_input_shape(self, shape):
        super(ConvLayerCUDA, self).set_input_shape(shape)

    def get_output_shape(self):
        return super(ConvLayerCUDA, self).get_output_shape()

    def update_parameters(self, learning_rate):
        mod = SourceModule("""
            #define L_RATE """ + str(learning_rate) + """

            __global__ void param_update(float *f_weights, float *biases, float *d_f_weights,
                                         float *d_biases) {
                if (!threadIdx.x)
                    biases[blockIdx.x] -= L_RATE * d_biases[blockIdx.x];
                    d_biases[blockIdx.x] = 0.0;

                int idx = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x +
                          threadIdx.x;
                f_weights[idx] -= L_RATE * d_f_weights[idx];
                d_f_weights[idx] = 0.0;
            }
            """)
        param_update = mod.get_function('param_update')
        param_update(driver.InOut(self.filter_weights), driver.InOut(self.biases),
                     driver.InOut(self.d_filter_weights), driver.InOut(self.d_biases),
                     block=(self.filter_shape[1], self.filter_shape[1], 1),
                     grid=(self.num_filters, 1, 1))

if __name__ == '__main__':
    dummy_input = np.ones((128, 599))
    # print "Input:\n", dummy_input

    layer = ConvLayerCUDA(num_filters=64, filter_shape=(128, 4), weight_decay=0, weight_scale=0.01,
                          padding_mode=False)
    layer.set_input_shape((128, 599))

    start = time.time()
    # print "\n--->> Forward propagation:\n",
    layer.forward_prop(dummy_input)
    finish = time.time()
    print "Fwd prop - time taken: ", finish - start

    dummy_output_grad = np.ones((64, 596)) / 2
    print "\nOutput gradient:\n", dummy_output_grad

    print "\n--->> Backpropagation:\n", layer.back_prop(dummy_output_grad)

    start = time.time()
    # print "\n--->> Params before update:\n", layer.filter_weights, "\n", layer.biases
    layer.update_parameters(0.01)
    # print "\n--->> Params after update:\n", layer.filter_weights, "\n", layer.biases
    finish = time.time()
    print "Param update - time taken: ", finish - start
