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
            const int INPUT_LENGTH = """ + str(self.input_shape[1]) + """;

            __global__ void global_pool(float *in, float *out, float *max_idx) {
                __shared__ float temp[INPUT_LENGTH];
                temp[threadIdx.x] = in[threadIdx.x];

                __syncthreads();
                if (threadIdx.x)
                    return;

                int i_max = 0;
                float max_val = temp[0];
                float sum = 0.0;
                float sum_sqr = 0.0;

                for (int i = 0; i < blockDim.x; ++i) {
                    sum += temp[i];
                    sum_sqr += temp[i] * temp[i];
                    if (max_val < temp[i]) {
                        max_val = temp[i];
                        i_max = i;
                    }
                }

                out[blockIdx.x] = sum / (float)blockDim.x;
                out[blockIdx.x + gridDim.x] = max_val;
                max_idx[blockIdx.x] = i_max;
                out[blockIdx.x + (gridDim.x << 1)] = sqrt(sum_sqr);
            }
            """)
        global_pool = mod.get_function('global_pool')
        global_pool(driver.In(input.astype(np.float32)),
                    driver.InOut(output), driver.InOut(self.max_activation_indices),
                    block=(self.input_shape[1], 1, 1), grid=(self.input_shape[0], 1, 1))

        return output

    def back_prop(self, output_grad):
        raise NotImplementedError()

    def set_input_shape(self, shape):
        super(GlobalPoolingLayerCUDA, self).set_input_shape(shape)

    def get_output_shape(self):
        return super(GlobalPoolingLayerCUDA, self).get_output_shape()

if __name__ == '__main__':
    dummy_input = np.ones((64, 70))
    # print "Input:\n", dummy_input

    layer = GlobalPoolingLayerCUDA()
    layer.set_input_shape(dummy_input.shape)

    start = time.time()
    # print "Forward propagation:\n",
    layer.forward_prop(dummy_input)
    finish = time.time()
    print "Fwd prop - time taken: ", finish - start
