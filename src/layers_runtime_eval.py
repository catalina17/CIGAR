import matplotlib.pyplot as pyplot
import numpy as np
import os
import time

from convnet_layers.activation_layer import ActivationLayer
from convnet_layers.activation_layer_cuda import ActivationLayerCUDA
from convnet_layers.conv_layer import ConvLayer
from convnet_layers.conv_layer_cuda import ConvLayerCUDA
from convnet_layers.fullyconnected_layer import FullyConnectedLayer
from convnet_layers.fullyconnected_layer_cuda import FullyConnectedLayerCUDA
from convnet_layers.globalpooling_layer import GlobalPoolingLayer
from convnet_layers.globalpooling_layer_cuda import GlobalPoolingLayerCUDA
from convnet_layers.maxpooling_layer import MaxPoolingLayer
from convnet_layers.maxpooling_layer_cuda import MaxPoolingLayerCUDA


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


def compute_statistics():
    layer_types = ['activation', 'global_pooling', 'max_pooling', 'fully_connected', 'conv']

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


def read_statistics(dir_path, start_pow_2):
    f_cpu_fwdprop = open(dir_path + 'cpu_fwdprop.txt', 'r')
    f_gpu_fwdprop = open(dir_path + 'gpu_fwdprop.txt', 'r')
    f_cpu_backprop = open(dir_path + 'cpu_backprop.txt', 'r')
    f_gpu_backprop = open(dir_path + 'gpu_backprop.txt', 'r')

    data_cpu_fwd_prop = np.array(f_cpu_fwdprop.readlines())
    data_cpu_back_prop = np.array(f_cpu_backprop.readlines())
    data_gpu_fwd_prop = np.array(f_gpu_fwdprop.readlines())
    data_gpu_back_prop = np.array(f_gpu_backprop.readlines())

    f_cpu_fwdprop.close()
    f_gpu_fwdprop.close()
    f_cpu_backprop.close()
    f_gpu_backprop.close()

    runtimes_cpu_fwd_prop = np.empty((data_cpu_fwd_prop.shape[0] / 11))
    runtimes_cpu_fwd_prop_sigmas = np.empty((data_cpu_fwd_prop.shape[0] / 11))
    runtimes_cpu_back_prop = np.empty((data_cpu_back_prop.shape[0] / 11))
    runtimes_cpu_back_prop_sigmas = np.empty((data_cpu_back_prop.shape[0] / 11))
    runtimes_gpu_fwd_prop = np.empty((data_gpu_fwd_prop.shape[0] / 11))
    runtimes_gpu_fwd_prop_sigmas = np.empty((data_gpu_fwd_prop.shape[0] / 11))
    runtimes_gpu_back_prop = np.empty((data_gpu_back_prop.shape[0] / 11))
    runtimes_gpu_back_prop_sigmas = np.empty((data_gpu_back_prop.shape[0] / 11))

    current_pow_2 = start_pow_2
    float_runtimes_i = np.empty((9, ), dtype=float)

    for i in range(data_cpu_fwd_prop.shape[0] / 11):

        # CPU Forward Propagation
        runtimes_i = data_cpu_fwd_prop[i * 11 : (i + 1) * 11]
        for run in range(2, 11):
            float_runtimes_i[run - 2] = float(runtimes_i[run])
        runtimes_cpu_fwd_prop[i] = np.mean(float_runtimes_i)
        runtimes_cpu_fwd_prop_sigmas[i] = np.std(float_runtimes_i)

        # CPU Back Propagation
        runtimes_i = data_cpu_back_prop[i * 11 : (i + 1) * 11]
        for run in range(2, 11):
            float_runtimes_i[run - 2] = float(runtimes_i[run])
        runtimes_cpu_back_prop[i] = np.mean(float_runtimes_i)
        runtimes_cpu_back_prop_sigmas[i] = np.std(float_runtimes_i)

        # GPU Forward Propagation
        runtimes_i = data_gpu_fwd_prop[i * 11 : (i + 1) * 11]
        for run in range(2, 11):
            float_runtimes_i[run - 2] = float(runtimes_i[run])
        runtimes_gpu_fwd_prop[i] = np.mean(float_runtimes_i)
        runtimes_gpu_fwd_prop_sigmas[i] = np.std(float_runtimes_i)

        # CPU Back Propagation
        runtimes_i = data_gpu_back_prop[i * 11 : (i + 1) * 11]
        for run in range(2, 11):
            float_runtimes_i[run - 2] = float(runtimes_i[run])
        runtimes_gpu_back_prop[i] = np.mean(float_runtimes_i)
        runtimes_gpu_back_prop_sigmas[i] = np.std(float_runtimes_i)

        current_pow_2 += 1

    return {
        'cpu_fwd': runtimes_cpu_fwd_prop,
        'cpu_fwd_sigmas': runtimes_cpu_fwd_prop_sigmas,
        'cpu_back': runtimes_cpu_back_prop,
        'cpu_back_sigmas': runtimes_cpu_back_prop_sigmas,
        'gpu_fwd': runtimes_gpu_fwd_prop,
        'gpu_fwd_sigmas': runtimes_gpu_fwd_prop_sigmas,
        'gpu_back': runtimes_gpu_back_prop,
        'gpu_back_sigmas': runtimes_gpu_back_prop_sigmas,
    }


def plot_statistics():
    layer_types = ['activation', 'conv', 'fully_connected', 'global_pooling', 'max_pooling']

    for layer_type in layer_types:
        # Paths for saving plots
        path_cata = './layers_eval_cata/' + layer_type + '/'
        path_profir = './layers_eval_profir/' + layer_type + '/'

        # Starting power of 2 of the input size for the current type of layer
        if layer_type == 'activation':
            start_pow = 1
        elif layer_type == 'conv':
            start_pow = 5
        elif layer_type == 'fully_connected':
            start_pow = 1
        elif layer_type == 'global_pooling':
            start_pow = 1
        elif layer_type == 'max_pooling':
            start_pow = 7

        # Load stats from file
        statistics_cata = read_statistics(path_cata, start_pow)
        statistics_profir = read_statistics(path_profir, start_pow)

        # x axis values for log2(input_size)
        x_cata = np.array(range(start_pow, statistics_cata['cpu_fwd'].shape[0] + start_pow))
        x_profir = np.array(range(start_pow, statistics_profir['cpu_fwd'].shape[0] + start_pow))

        cpu_fwd_cata = statistics_cata['cpu_fwd']
        cpu_fwd_cata_sigmas = statistics_cata['cpu_fwd_sigmas']
        cpu_back_cata = statistics_cata['cpu_back']
        cpu_back_cata_sigmas = statistics_cata['cpu_back_sigmas']
        gpu_fwd_cata = statistics_cata['gpu_fwd']
        gpu_fwd_cata_sigmas = statistics_cata['gpu_fwd_sigmas']
        gpu_back_cata = statistics_cata['gpu_back']
        gpu_back_cata_sigmas = statistics_cata['gpu_back_sigmas']
        gpu_fwd_profir = statistics_profir['gpu_fwd']
        gpu_fwd_profir_sigmas = statistics_profir['gpu_fwd_sigmas']
        gpu_back_profir = statistics_profir['gpu_back']
        gpu_back_profir_sigmas = statistics_profir['gpu_back_sigmas']

        # Plot runtimes for the forward propagation operation
        lim = min(x_cata.shape[0], x_profir.shape[0])
        x = x_cata[:lim]
        pyplot.errorbar(x, cpu_fwd_cata[:lim], yerr=cpu_fwd_cata_sigmas[:lim],
                        label='CPU')
        pyplot.errorbar(x, gpu_fwd_cata[:lim], yerr=gpu_fwd_cata_sigmas[:lim],
                        label='GT640M')
        pyplot.errorbar(x, gpu_fwd_profir[:lim], yerr=gpu_fwd_profir_sigmas[:lim],
                        label='GTX980M')
        pyplot.legend(loc='lower right')
        pyplot.yscale('log')
        pyplot.grid(True, which='both', linestyle='solid', color='0.8')
        pyplot.xlabel('Base-2 logarithm of input size')
        pyplot.ylabel('Forward propagation running time (s)')
        pyplot.show()
        pyplot.savefig(layer_type + '_fwd_prop.pdf', format='pdf')

        pyplot.clf()

        # Plot runtimes for the forward propagation operation
        pyplot.errorbar(x, cpu_back_cata[:lim], yerr=cpu_back_cata_sigmas[:lim],
                        label='CPU')
        pyplot.errorbar(x, gpu_back_cata[:lim], yerr=gpu_back_cata_sigmas[:lim],
                        label='GT640M')
        pyplot.errorbar(x, gpu_back_profir[:lim], yerr=gpu_back_profir_sigmas[:lim],
                        label='GTX980M')
        pyplot.legend(loc='lower right')
        pyplot.yscale('log')
        pyplot.grid(True, which='both', linestyle='solid', color='0.8')
        pyplot.xlabel('Base-2 logarithm of gradient size')
        pyplot.ylabel('Back-propagation running time (s)')
        pyplot.show()
        pyplot.savefig(layer_type + '_back_prop.pdf', format='pdf')

        pyplot.clf()

if __name__ == '__main__':
    plot_statistics()
