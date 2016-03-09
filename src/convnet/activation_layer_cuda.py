from activation_layer import ActivationLayer
from pycuda.compiler import SourceModule

import pycuda.driver as driver
import pycuda.autoinit
import numpy as np
import time


class ActivationLayerCUDA(ActivationLayer):

    def __init__(self, activation_fn):
        mod = SourceModule("""
            __global__ void multiply_them(float *dest, float *a, float *b) {
                const int i = threadIdx.x;
                dest[i] = a[i] * b[i];
            }
            """)
        super(ActivationLayerCUDA, self).__init__(activation_fn)

    def forward_prop(self, input):
        assert self._input_shape == input.shape, "Input does not have correct shape"
        self._current_input = input

        h = None
        w = None
        if len(self._input_shape) == 1:
            h = 1
            w = self._input_shape[0]
        else:
            h, w = self._input_shape

        mod = SourceModule("""
            #define INPUT_HEIGHT """ + str(h) + """
            #define INPUT_WIDTH """ + str(w) + """

            __global__ void fwd_leakyrelu(float *in, float *out) {
                int col = blockIdx.x * blockDim.x + threadIdx.x;
                int row = blockIdx.y * blockDim.y + threadIdx.y;

                if (col >= INPUT_WIDTH || row >= INPUT_HEIGHT)
                    return;

                int idx = col + row * INPUT_WIDTH;
                out[idx] = in[idx];
                if (in[idx] <= 0.0)
                    out[idx] *= 0.01;
            }
            """)

        fwd_leakyrelu = mod.get_function('fwd_leakyrelu')
        output = np.empty(self._input_shape).astype(np.float32)
        fwd_leakyrelu(driver.In(input.astype(np.float32)),
                      driver.Out(output),
                      block=(8, 4, 1), grid=(w / 8 + 1, h / 4 + 1, 1))

        return output

    def back_prop(self, output_grad):
        h = None
        w = None
        if len(self._input_shape) == 1:
            h = 1
            w = self._input_shape[0]
        else:
            h, w = self._input_shape

        mod = SourceModule("""
            #define INPUT_HEIGHT """ + str(h) + """
            #define INPUT_WIDTH """ + str(w) + """

            __global__ void backprop_leakyrelu(float *grad_out, float *grad_in, float *in) {
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
        input_grad = np.empty(self._input_shape).astype(np.float32)
        backprop_leakyrelu(driver.In(output_grad.astype(np.float32)),
                           driver.Out(input_grad),
                           driver.In(self._current_input.astype(np.float32)),
                           block=(8, 4, 1), grid=(w / 8 + 1, h / 4 + 1, 1))

        return input_grad

    def set_input_shape(self, shape):
        super(ActivationLayerCUDA, self).set_input_shape(shape)

    def get_output_shape(self):
        return super(ActivationLayerCUDA, self).get_output_shape()

if __name__ == '__main__':
    dummy_input = np.random.randn(64, 596)
    print "Input:\n", dummy_input

    layer = ActivationLayerCUDA('leakyReLU')
    layer.set_input_shape((64, 596))

    start = time.time()
    print "\n--->> Forward propagation:\n", layer.forward_prop(dummy_input)
    finish = time.time()
    print "Fwd prop - Time taken: ", finish - start

    dummy_output_grad = np.ones((64, 596))
    print "Output gradient:\n", dummy_output_grad

    start = time.time()
    print "\n--->> Backpropagation:\n", layer.back_prop(dummy_output_grad)
    finish = time.time()
    print "Back prop - Time taken: ", finish - start
