from globalpooling_layer import GlobalPoolingLayer
from pycuda.compiler import SourceModule

import pycuda.driver as driver
import pycuda.autoinit
import numpy as np
import time


class GlobalPoolingLayerCUDA(GlobalPoolingLayer):

    def __init__(self):
        mod = SourceModule("""
            __global__ void multiply_them(float *dest, float *a, float *b) {
                const int i = threadIdx.x;
                dest[i] = a[i] * b[i];
            }
            """)
        super(GlobalPoolingLayerCUDA, self).__init__()

    def forward_prop(self, input):
        assert self.input_shape == input.shape, "Input does not have correct shape"
        self.current_input = input

        output = np.empty(self.get_output_shape()).astype(np.float32)
        self.max_activation_indices = np.empty(self.input_shape).astype(np.float32)

        mod = SourceModule("""
            __global__ void global_pool(float *in, float *out, float *max_idx) {
            }
            """)
        global_pool = mod.get_function('global_pool')
        global_pool(driver.In(input.astype(np.float32)),
                    driver.InOut(output), driver.InOut(self.max_activation_indices),
                    block=(self.input_shape[1], 1, 1), grid=(self.input_shape[0], 1, 1))

    def back_prop(self, output_grad):
        raise NotImplementedError()

    def set_input_shape(self, shape):
        super(GlobalPoolingLayerCUDA, self).set_input_shape(shape)

    def get_output_shape(self):
        return super(GlobalPoolingLayerCUDA, self).get_output_shape()
