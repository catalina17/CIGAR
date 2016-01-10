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
        self.current_input = input.astype(np.float32)

        mod = SourceModule("""
            #define INPUT_LENGTH """ + str(self.input_shape[0]) + """
            #define OUTPUT_LENGTH """ + str(self.num_nodes) + """

            __global__ void fwd_prop(float *in, float *weights, float *biases, float *out) {
                __shared__ float temp[INPUT_LENGTH];
                temp[threadIdx.x] = in[threadIdx.x] *
                                    weights[OUTPUT_LENGTH * threadIdx.x + blockIdx.x];

                __syncthreads();
                if (threadIdx.x)
                    return;

                for (int i = 0; i < INPUT_LENGTH; ++i)
                    out[blockIdx.x] += temp[i];

                out[blockIdx.x] += biases[blockIdx.x];
            }
            """)

        fwd_prop = mod.get_function('fwd_prop')
        output = np.zeros(self.num_nodes).astype(np.float32)
        fwd_prop(driver.In(input.astype(np.float32)), driver.In(self.weights.astype(np.float32)),
                 driver.In(self.biases.astype(np.float32)), driver.InOut(output),
                 block=(self.input_shape[0], 1, 1), grid=(self.num_nodes, 1, 1))

        return output

    def back_prop(self, output_grad):
        # No weight decay

        mod = SourceModule("""
            #define INPUT_LENGTH """ + str(self.input_shape[0]) + """
            #define OUTPUT_LENGTH """ + str(self.num_nodes) + """

            __global__ void back_prop(float *out_grad, float *in_grad, float *d_weights,
                                      float *d_biases, float *in, float *weights) {
                __shared__ float temp[OUTPUT_LENGTH];
                temp[threadIdx.x] = out_grad[threadIdx.x] *
                                    weights[INPUT_LENGTH * threadIdx.x + blockIdx.x];

                if (!blockIdx.x)
                    d_biases[threadIdx.x] += out_grad[threadIdx.x];

                d_weights[OUTPUT_LENGTH * blockIdx.x + threadIdx.x] += in[blockIdx.x] *
                                                                       out_grad[threadIdx.x];

                __syncthreads();
                if (threadIdx.x)
                    return;
                for (int i = 0; i < OUTPUT_LENGTH; ++i)
                    in_grad[blockIdx.x] += temp[i];
            }
            """)
        back_prop = mod.get_function('back_prop')
        input_grad = np.zeros(self.input_shape).astype(np.float32)
        back_prop(driver.In(output_grad.astype(np.float32)),
                  driver.InOut(input_grad),
                  driver.InOut(self.d_weights),
                  driver.InOut(self.d_biases),
                  driver.In(self.current_input),
                  driver.In(self.weights.T),
                  block=(self.num_nodes, 1, 1), grid=(self.input_shape[0], 1, 1))

        return input_grad

    def set_input_shape(self, shape):
        super(FullyConnectedLayerCUDA, self).set_input_shape(shape)

    def get_output_shape(self):
        return super(FullyConnectedLayerCUDA, self).get_output_shape()

    def update_parameters(self, learning_rate):
        mod = SourceModule("""
            #define L_RATE """ + str(learning_rate) + """
            #define INPUT_LENGTH """ + str(self.input_shape[0]) + """
            #define OUTPUT_LENGTH """ + str(self.num_nodes) + """

            __global__ void param_update(float *d_weights, float *d_biases, float *weights,
                                         float *biases) {
                if (!blockIdx.x) {
                    biases[threadIdx.x] -= L_RATE * d_biases[threadIdx.x];
                    d_biases[threadIdx.x] = 0.0;
                }

                int idx = blockIdx.x * OUTPUT_LENGTH + threadIdx.x;
                weights[idx] -= L_RATE * d_weights[idx];
                d_weights[idx] = 0.0;
            }
            """)

        param_update = mod.get_function('param_update')
        param_update(driver.InOut(self.d_weights),
                     driver.InOut(self.d_biases),
                     driver.InOut(self.weights),
                     driver.InOut(self.biases),
                     block=(self.num_nodes, 1, 1), grid=(self.input_shape[0], 1, 1))

if __name__ == "__main__":
    dummy_input = np.ones((192,))
    # print "Input:\n", dummy_input

    layer = FullyConnectedLayerCUDA(64, 0, 0.01)
    layer.set_input_shape((192,))

    start = time.time()
    # print "\n--->> Forward propagation:\n", sum(layer.weights), "\n",
    layer.forward_prop(dummy_input)
    finish = time.time()
    print "Fwd prop - Time taken: ", finish - start

    dummy_output_grad = np.random.randn(64)
    # print "\nOutput gradient:\n", dummy_output_grad

    start = time.time()
    # print "\n--->> Backpropagation:\n",
    layer.back_prop(dummy_output_grad)
    finish = time.time()
    print "Back prop - Time taken: ", finish - start

    start = time.time()
    # print "\n--->> Params before update:\n", layer.weights, "\n", layer.biases
    layer.update_parameters(0.01)
    # print "\n--->> Params after update:\n", layer.weights, "\n", layer.biases
    finish = time.time()
    print "Param update - Time taken: ", finish - start
