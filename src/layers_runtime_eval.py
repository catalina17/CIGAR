import numpy as np
import os
import time

from convnet.activation_layer import ActivationLayer
from convnet.activation_layer_cuda import ActivationLayerCUDA
from convnet.conv_layer import ConvLayer
from convnet.conv_layer_cuda import ConvLayerCUDA
from convnet.fullyconnected_layer import FullyConnectedLayer
from convnet.fullyconnected_layer_cuda import FullyConnectedLayerCUDA
from convnet.globalpooling_layer import GlobalPoolingLayer
from convnet.globalpooling_layer_cuda import GlobalPoolingLayerCUDA
from convnet.maxpooling_layer import MaxPoolingLayer
from convnet.maxpooling_layer_cuda import MaxPoolingLayerCUDA


def activation(dir_path, runs):
    print "-----------> Activation \n"

    f_cpu_fwdprop = open(dir_path + 'cpu_fwdprop.txt', 'w')
    f_gpu_fwdprop = open(dir_path + 'gpu_fwdprop.txt', 'w')
    f_cpu_backprop = open(dir_path + 'cpu_backprop.txt', 'w')
    f_gpu_backprop = open(dir_path + 'gpu_backprop.txt', 'w')

    layer = ActivationLayer('leakyReLU')
    layer_cuda = ActivationLayerCUDA('leakyReLU')

    width = 1
    height = 1
    for power in range(25):
        if power % 2 == 0:
            width *= 2
        else:
            height *= 2

        shape = (height, width)

        info = "Input shape: (" + str(height) + "," + str(width) + ")\n"
        f_cpu_fwdprop.write(info)
        f_gpu_fwdprop.write(info)
        f_cpu_backprop.write(info)
        f_gpu_backprop.write(info)
        print info

        layer.set_input_shape(shape)
        layer_cuda.set_input_shape(shape)

        input = np.random.randn(height, width)
        output = np.random.randn(height, width)

        for run in range(runs):
            # Forward prop --- CPU
            start = time.time()
            layer.forward_prop(input)
            finish = time.time()
            time_cpu = finish - start
            f_cpu_fwdprop.write(str(time_cpu) + "\n")

            # Forward prop --- GPU
            start = time.time()
            layer_cuda.forward_prop(input)
            finish = time.time()
            time_cuda = finish - start
            f_gpu_fwdprop.write(str(time_cuda) + "\n")

            # Back prop --- CPU
            start = time.time()
            layer.back_prop(output)
            finish = time.time()
            time_cpu = finish - start
            f_cpu_backprop.write(str(time_cpu) + "\n")

            # Back prop --- GPU
            start = time.time()
            layer_cuda.back_prop(output)
            finish = time.time()
            time_cuda = finish - start
            f_gpu_backprop.write(str(time_cuda) + "\n")

    f_cpu_fwdprop.close()
    f_gpu_fwdprop.close()
    f_cpu_backprop.close()
    f_gpu_backprop.close()


def conv(dir_path, runs):
    print "-----------> Convolutional \n"

    f_cpu_fwdprop = open(dir_path + 'cpu_fwdprop.txt', 'w')
    f_gpu_fwdprop = open(dir_path + 'gpu_fwdprop.txt', 'w')
    f_cpu_backprop = open(dir_path + 'cpu_backprop.txt', 'w')
    f_gpu_backprop = open(dir_path + 'gpu_backprop.txt', 'w')

    filter_width = 4
    filter_height = 4
    input_width = 4
    input_height = 4
    num_filters = 32
    for power in range(20):
        if power % 2 == 1 and filter_width * filter_height < 512:
            filter_height *= 2
            input_height *= 2
        else:
            input_width *= 2

        info = "(Input, filter): (" + str(input_height) + ", " + str(input_width) +\
               "), (" + str(filter_height) + ", " + str(filter_width) + ")" + "\n"
        f_cpu_fwdprop.write(info)
        f_gpu_fwdprop.write(info)
        f_cpu_backprop.write(info)
        f_gpu_backprop.write(info)
        print info

        layer = ConvLayer(num_filters, filter_shape=(filter_height, filter_width), weight_scale=0.1,
                          padding_mode=False)
        layer_cuda = ConvLayerCUDA(num_filters, filter_shape=(filter_height, filter_width),
                                   weight_scale=0.1, padding_mode=False)

        input_shape = (input_height, input_width)
        layer.set_input_shape(input_shape)
        layer_cuda.set_input_shape(input_shape)

        input = np.random.randn(input_shape[0], input_shape[1])
        output = np.random.randn(layer.get_output_shape()[0], layer.get_output_shape()[1])

        for run in range(runs):
            # Forward prop --- CPU
            start = time.time()
            layer.forward_prop(input)
            finish = time.time()
            time_cpu = finish - start
            f_cpu_fwdprop.write(str(time_cpu) + "\n")

            # Forward prop --- GPU
            start = time.time()
            layer_cuda.forward_prop(input)
            finish = time.time()
            time_cuda = finish - start
            f_gpu_fwdprop.write(str(time_cuda) + "\n")

            # Back prop --- CPU
            start = time.time()
            layer.back_prop(output)
            finish = time.time()
            time_cpu = finish - start
            f_cpu_backprop.write(str(time_cpu) + "\n")

            # Back prop --- GPU
            start = time.time()
            layer_cuda.back_prop(output)
            finish = time.time()
            time_cuda = finish - start
            f_gpu_backprop.write(str(time_cuda) + "\n")

    f_cpu_fwdprop.close()
    f_gpu_fwdprop.close()
    f_cpu_backprop.close()
    f_gpu_backprop.close()


def fully_connected(dir_path, runs):
    print "-----------> Fully-Connected \n"

    f_cpu_fwdprop = open(dir_path + 'cpu_fwdprop.txt', 'w')
    f_gpu_fwdprop = open(dir_path + 'gpu_fwdprop.txt', 'w')
    f_cpu_backprop = open(dir_path + 'cpu_backprop.txt', 'w')
    f_gpu_backprop = open(dir_path + 'gpu_backprop.txt', 'w')

    input_length = 1
    num_nodes = 1
    for power in range(18):
        print power
        if power % 2 == 0:
            input_length *= 2
        else:
            num_nodes *= 2

        info = "(Input length, nodes): (" + str(input_length) + "," + str(num_nodes) + ")\n"
        f_cpu_fwdprop.write(info)
        f_gpu_fwdprop.write(info)
        f_cpu_backprop.write(info)
        f_gpu_backprop.write(info)
        print info

        layer = FullyConnectedLayer(num_nodes, 0.2)
        layer_cuda = FullyConnectedLayerCUDA(num_nodes, 0.2)

        layer.set_input_shape((input_length, ))
        layer_cuda.set_input_shape((input_length, ))

        input = np.random.randn(input_length)
        output = np.random.randn(num_nodes)

        for run in range(runs):
            # Forward prop --- CPU
            start = time.time()
            layer.forward_prop(input)
            finish = time.time()
            time_cpu = finish - start
            f_cpu_fwdprop.write(str(time_cpu) + "\n")

            # Forward prop --- GPU
            start = time.time()
            layer_cuda.forward_prop(input)
            finish = time.time()
            time_cuda = finish - start
            f_gpu_fwdprop.write(str(time_cuda) + "\n")

            # Back prop --- CPU
            start = time.time()
            layer.back_prop(output)
            finish = time.time()
            time_cpu = finish - start
            f_cpu_backprop.write(str(time_cpu) + "\n")

            # Back prop --- GPU
            start = time.time()
            layer_cuda.back_prop(output)
            finish = time.time()
            time_cuda = finish - start
            f_gpu_backprop.write(str(time_cuda) + "\n")

    f_cpu_fwdprop.close()
    f_gpu_fwdprop.close()
    f_cpu_backprop.close()
    f_gpu_backprop.close()


def global_pooling(dir_path, runs):
    print "-----------> Global Pooling \n"

    f_cpu_fwdprop = open(dir_path + 'cpu_fwdprop.txt', 'w')
    f_gpu_fwdprop = open(dir_path + 'gpu_fwdprop.txt', 'w')
    f_cpu_backprop = open(dir_path + 'cpu_backprop.txt', 'w')
    f_gpu_backprop = open(dir_path + 'gpu_backprop.txt', 'w')

    layer = GlobalPoolingLayer()
    layer_cuda = GlobalPoolingLayerCUDA()

    width = 1
    height = 1
    for power in range(25):
        if power % 2 == 0:
            width *= 2
        else:
            height *= 2

        shape = (height, width)

        info = "Input shape: (" + str(height) + "," + str(width) + ")\n"
        f_cpu_fwdprop.write(info)
        f_gpu_fwdprop.write(info)
        f_cpu_backprop.write(info)
        f_gpu_backprop.write(info)
        print info

        layer.set_input_shape(shape)
        layer_cuda.set_input_shape(shape)

        input = np.random.randn(height, width)
        output = np.random.randn(3 * height)

        for run in range(runs):
            # Forward prop --- CPU
            start = time.time()
            layer.forward_prop(input)
            finish = time.time()
            time_cpu = finish - start
            f_cpu_fwdprop.write(str(time_cpu) + "\n")

            # Forward prop --- GPU
            start = time.time()
            layer_cuda.forward_prop(input)
            finish = time.time()
            time_cuda = finish - start
            f_gpu_fwdprop.write(str(time_cuda) + "\n")

            # Back prop --- CPU
            start = time.time()
            layer.back_prop(output)
            finish = time.time()
            time_cpu = finish - start
            f_cpu_backprop.write(str(time_cpu) + "\n")

            # Back prop --- GPU
            start = time.time()
            layer_cuda.back_prop(output)
            finish = time.time()
            time_cuda = finish - start
            f_gpu_backprop.write(str(time_cuda) + "\n")

    f_cpu_fwdprop.close()
    f_gpu_fwdprop.close()
    f_cpu_backprop.close()
    f_gpu_backprop.close()


def max_pooling(dir_path, runs):
    print "-----------> Max Pooling \n"

    f_cpu_fwdprop = open(dir_path + 'cpu_fwdprop.txt', 'w')
    f_gpu_fwdprop = open(dir_path + 'gpu_fwdprop.txt', 'w')
    f_cpu_backprop = open(dir_path + 'cpu_backprop.txt', 'w')
    f_gpu_backprop = open(dir_path + 'gpu_backprop.txt', 'w')

    layer = MaxPoolingLayer((2, 2))
    layer_cuda = MaxPoolingLayerCUDA((2, 2))

    width = 8
    height = 8
    for power in range(20):
        if power % 2 == 0:
            width *= 2
        else:
            height *= 2

        shape = (height, width)
        print shape

        info = "Input shape: (" + str(height) + "," + str(width) + ")\n"
        f_cpu_fwdprop.write(info)
        f_gpu_fwdprop.write(info)
        f_cpu_backprop.write(info)
        f_gpu_backprop.write(info)
        print info

        layer.set_input_shape(shape)
        layer_cuda.set_input_shape(shape)

        input = np.random.randn(height, width)
        output = np.random.randn(height / 2, width / 2)

        for run in range(runs):
            # Forward prop --- CPU
            start = time.time()
            layer.forward_prop(input)
            finish = time.time()
            time_cpu = finish - start
            f_cpu_fwdprop.write(str(time_cpu) + "\n")

            # Forward prop --- GPU
            start = time.time()
            layer_cuda.forward_prop(input)
            finish = time.time()
            time_cuda = finish - start
            f_gpu_fwdprop.write(str(time_cuda) + "\n")

            # Back prop --- CPU
            start = time.time()
            layer.back_prop(output)
            finish = time.time()
            time_cpu = finish - start
            f_cpu_backprop.write(str(time_cpu) + "\n")

            # Back prop --- GPU
            start = time.time()
            layer_cuda.back_prop(output)
            finish = time.time()
            time_cuda = finish - start
            f_gpu_backprop.write(str(time_cuda) + "\n")

    f_cpu_fwdprop.close()
    f_gpu_fwdprop.close()
    f_cpu_backprop.close()
    f_gpu_backprop.close()


if __name__ == '__main__':
    layer_types = ['activation', 'global_pooling', 'max_pooling', 'fully_connected', 'conv']
    layer_types = ['conv']

    for layer_type in layer_types:
        path = './layers_eval/' + layer_type + '/'
        if not os.path.exists(path):
            os.makedirs(path)

        run_count = 10

        if layer_type == 'activation':
            activation(path, run_count)
        elif layer_type == 'global_pooling':
            global_pooling(path, run_count)
        elif layer_type == 'max_pooling':
            max_pooling(path, run_count)
        elif layer_type == 'fully_connected':
            fully_connected(path, run_count)
        elif layer_type == 'conv':
            conv(path, run_count)
