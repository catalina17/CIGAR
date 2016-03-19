import numpy as np
import time

import pycuda.autoinit
import pycuda.driver as driver
from pycuda.compiler import SourceModule

from fullyconnected_layer import FullyConnectedLayer


class FullyConnectedLayerCUDA(FullyConnectedLayer):

    def __init__(self, num_nodes, weight_scale):
        mod = SourceModule("""
            __global__ void multiply_them(float *dest, float *a, float *b) {
                const int i = threadIdx.x;
                dest[i] = a[i] * b[i];
            }
            """)
        super(FullyConnectedLayerCUDA, self).__init__(num_nodes, weight_scale)

    def forward_prop(self, input):
        assert self._input_shape == input.shape, "Input does not have correct shape"
        self._current_input = input.astype(np.float64)

        mod = SourceModule("""
            #define INPUT_LENGTH """ + str(self._input_shape[0]) + """

            __global__ void fwd_prop(double *in, double *weights, double *biases, double *out) {
                __shared__ double temp[INPUT_LENGTH];
                temp[threadIdx.x] = in[threadIdx.x] *
                                    weights[gridDim.x * threadIdx.x + blockIdx.x];

                __syncthreads();
                if (threadIdx.x)
                    return;

                for (int i = 0; i < blockDim.x; ++i)
                    out[blockIdx.x] += temp[i];

                out[blockIdx.x] += biases[blockIdx.x];
            }
            """)

        fwd_prop = mod.get_function('fwd_prop')
        output = np.zeros(self._num_nodes).astype(np.float64)
        fwd_prop(driver.In(input.astype(np.float64)), driver.In(self._weights.astype(np.float64)),
                 driver.In(self._biases.astype(np.float64)), driver.InOut(output),
                 block=(self._input_shape[0], 1, 1), grid=(self._num_nodes, 1, 1))

        return output

    def back_prop(self, output_grad):
        mod = SourceModule("""
            #define OUTPUT_LENGTH """ + str(self._num_nodes) + """

            __global__ void back_prop(double *out_grad, double *in_grad, double *d_weights,
                                      double *d_biases, double *in, double *weights) {
                __shared__ double temp[OUTPUT_LENGTH];
                temp[threadIdx.x] = out_grad[threadIdx.x] *
                                    weights[gridDim.x * threadIdx.x + blockIdx.x];

                if (!blockIdx.x)
                    d_biases[threadIdx.x] += out_grad[threadIdx.x];

                d_weights[blockDim.x * blockIdx.x + threadIdx.x] += in[blockIdx.x] *
                                                                    out_grad[threadIdx.x];

                __syncthreads();
                if (threadIdx.x)
                    return;
                for (int i = 0; i < blockDim.x; ++i)
                    in_grad[blockIdx.x] += temp[i];
            }
            """)
        back_prop = mod.get_function('back_prop')
        input_grad = np.zeros(self._input_shape).astype(np.float64)
        back_prop(driver.In(output_grad.astype(np.float64)),
                  driver.InOut(input_grad),
                  driver.InOut(self._d_weights),
                  driver.InOut(self._d_biases),
                  driver.In(self._current_input),
                  driver.In(self._weights.T),
                  block=(self._num_nodes, 1, 1), grid=(self._input_shape[0], 1, 1))

        return input_grad

    def set_input_shape(self, shape):
        super(FullyConnectedLayerCUDA, self).set_input_shape(shape)

    def get_output_shape(self):
        return super(FullyConnectedLayerCUDA, self).get_output_shape()

    def update_parameters(self, learning_rate):
        super(FullyConnectedLayerCUDA, self).update_parameters(learning_rate)

    def serialise_parameters(self, file_idx):
        super(FullyConnectedLayerCUDA, self).serialise_parameters(file_idx)

    def init_parameters_from_file(self, file_idx):
        super(FullyConnectedLayerCUDA, self).init_parameters_from_file(file_idx)

if __name__ == "__main__":
    dummy_input = np.ones((1024,))
    # print "Input:\n", dummy_input

    layer = FullyConnectedLayerCUDA(512, 0.01)
    layer.set_input_shape((1024,))

    start = time.time()
    # print "\n--->> Forward propagation:\n", sum(layer._weights), "\n",
    layer.forward_prop(dummy_input)
    finish = time.time()
    print "Fwd prop - Time taken: ", finish - start

    dummy_output_grad = np.random.randn(512)
    # print "\nOutput gradient:\n", dummy_output_grad

    start = time.time()
    # print "\n--->> Backpropagation:\n",
    layer.back_prop(dummy_output_grad)
    finish = time.time()
    print "Back prop - Time taken: ", finish - start
