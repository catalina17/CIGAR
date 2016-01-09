from fullyconnected_layer import FullyConnectedLayer
from pycuda.compiler import SourceModule

import pycuda.driver as driver
import pycuda.autoinit
import numpy as np
import time


class FullyConnectedLayerCUDA(FullyConnectedLayer):

    def __init__(self, num_nodes, weight_decay, weight_scale):
        mod = SourceModule("""
            __global__ void multiply_them(float *dest, float *a, float *b) {
                const int i = threadIdx.x;
                dest[i] = a[i] * b[i];
            }
            """)
        super(FullyConnectedLayerCUDA, self).__init__(num_nodes, weight_decay, weight_scale)

    def forward_prop(self, input):
        assert self.input_shape == input.shape, "Input does not have correct shape"
        self.current_input = input

        mod = SourceModule("""
            #define INPUT_LENGTH """ + str(self.input_shape[0]) + """
            #define OUTPUT_LENGTH """ + str(self.num_nodes) + """

            __global__ void fwd_prop(float *in, float *weights, float *biases, float *out) {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx >= OUTPUT_LENGTH)
                    return;

                float val = 0.0;
                for (int i = 0; i < INPUT_LENGTH; ++i)
                    val += in[i] * weights[idx + i * OUTPUT_LENGTH];
                val += biases[idx];

                out[idx] = val;
            }
            """)

        fwd_prop = mod.get_function('fwd_prop')
        output = np.empty(self.num_nodes).astype(np.float32)
        fwd_prop(driver.In(input.astype(np.float32)), driver.In(self.weights.astype(np.float32)),
                 driver.In(self.biases.astype(np.float32)), driver.Out(output), block=(32, 1, 1),
                 grid=(self.num_nodes / 32 + 1, 1, 1))

        return output

    def back_prop(self, output_grad):
        raise NotImplementedError()

    def set_input_shape(self, shape):
        super(FullyConnectedLayerCUDA, self).set_input_shape(shape)

    def get_output_shape(self):
        return super(FullyConnectedLayerCUDA, self).get_output_shape()

    def update_parameters(self, learning_rate):
        raise NotImplementedError()
