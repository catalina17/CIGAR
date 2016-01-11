from maxpooling_layer import MaxPoolingLayer
from pycuda.compiler import SourceModule

import pycuda.driver as driver
import pycuda.autoinit
import numpy as np
import time


class MaxPoolingLayerCUDA(MaxPoolingLayer):

    def __init__(self, filter_shape):
        mod = SourceModule("""
        __global__ void multiply_them(float *dest, float *a, float *b) {
            const int i = threadIdx.x;
            dest[i] = a[i] * b[i];
        }
        """)
        super(MaxPoolingLayerCUDA, self).__init__(filter_shape)

    def forward_prop(self, input):
        assert self.input_shape == input.shape, "Input does not have correct shape"
        self.current_input = input

        if (len(self.get_output_shape()) == 1):
            output = np.empty((1, self.get_output_shape()[0])).astype(np.float32)
        else:
            output = np.empty(self.get_output_shape()).astype(np.float32)

        self.max_activation_indices = np.empty(self.get_output_shape()).astype(np.int32)

        mod = None
        if self.filter_shape[1] == 4:
            mod = SourceModule("""
                #define INPUT_HEIGHT """ + str(self.input_shape[0]) + """
                #define INPUT_WIDTH """ + str(self.input_shape[1] / 4) + """

                __global__ void max_pool(float4 *in, float *dest, int *max_idx_w) {
                    int col = blockIdx.x * blockDim.x + threadIdx.x;
                    int row = blockIdx.y * blockDim.y;

                    if (col >= INPUT_WIDTH || row >= INPUT_HEIGHT)
                        return;

                    int idx = col + row * INPUT_WIDTH;
                    int start = col << 2;

                    float4 val = in[idx];
                    float max_val = val.x;
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
        elif self.filter_shape[1] == 2:
            mod = SourceModule("""
                #define INPUT_HEIGHT """ + str(self.input_shape[0]) + """
                #define INPUT_WIDTH """ + str(self.input_shape[1] / 2) + """

                __global__ void max_pool(float2 *in, float *dest, int *max_idx_w) {
                    int col = blockIdx.x * blockDim.x + threadIdx.x;
                    int row = blockIdx.y * blockDim.y;

                    if (col >= INPUT_WIDTH || row >= INPUT_HEIGHT)
                        return;

                    int idx = col + row * INPUT_WIDTH;
                    int start = col << 1;

                    float max_val = in[idx].x;
                    max_idx_w[idx] = start;

                    if (max_val < in[idx].y) {
                        max_val = in[idx].y;
                        max_idx_w[idx] = start + 1;
                    }

                    dest[idx] = max_val;
                }
                """)

        max_pool = mod.get_function('max_pool')
        max_pool(driver.In(input.astype(np.float32)), driver.Out(output),
                 driver.Out(self.max_activation_indices), block=(self.filter_shape[1], 1, 1),
                 grid=(self.get_output_shape()[1], self.get_output_shape()[0], 1))

        # print "OUT", output
        # print "Idx", self.max_activation_indices
        if output.shape[0] == 1:
            return output[0]
        else:
            return output

    def back_prop(self, output_grad):
        input_grad = np.zeros(self.input_shape).astype(np.float32)

        mod = SourceModule("""
            #define INGRAD_WIDTH """ + str(input_grad.shape[1]) + """

            __global__ void max_pool_back(float *outgrad, float *ingrad, int *max_idx_w) {
                int col = blockIdx.x * blockDim.x + threadIdx.x;
                int row = blockIdx.y * blockDim.y + threadIdx.y;

                if (col >= (gridDim.x << 3) || row >= (gridDim.y << 2))
                    return;

                int idx_out = col + row * (gridDim.x << 3);
                ingrad[max_idx_w[idx_out] + row * INGRAD_WIDTH] = outgrad[idx_out];
            }
            """)

        max_pool_back = mod.get_function('max_pool_back')
        max_pool_back(driver.In(output_grad.astype(np.float32)), driver.InOut(input_grad),
                      driver.In(self.max_activation_indices),
                      block=(8, 4, 1),
                      grid=(output_grad.shape[1] / 8, output_grad.shape[0] / 4, 1))

        return input_grad

    def set_input_shape(self, shape):
        super(MaxPoolingLayerCUDA, self).set_input_shape(shape)

    def get_output_shape(self):
        return super(MaxPoolingLayerCUDA, self).get_output_shape()

if __name__ == '__main__':
    layer = MaxPoolingLayerCUDA(filter_shape=(1, 4))
    input = np.random.randn(64, 596)
    layer.set_input_shape((64, 596))

    start = time.time()
    layer.forward_prop(input)

    output_grad = np.random.randn(64, 149)
    layer.back_prop(output_grad)
    finish = time.time()
    print "Time taken: ", finish - start
