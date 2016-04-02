import time

from convnet import ConvNet
from convnet_layers.activation_layer import ActivationLayer
from convnet_layers.conv_layer import ConvLayer
from convnet_layers.globalpooling_layer import GlobalPoolingLayer
from convnet_layers.fullyconnected_layer import FullyConnectedLayer
from convnet_layers.maxpooling_layer_cuda import MaxPoolingLayerCUDA
from convnet_layers.softmax_layer import SoftmaxLayer
from data_provider import DataProvider


def ten_class():
    neural_net = ConvNet([ConvLayer(32, (128, 4), weight_scale=0.044, padding_mode=False),
                          ActivationLayer('leakyReLU'),
                          MaxPoolingLayerCUDA((1, 4)),

                          ConvLayer(32, (32, 4), weight_scale=0.088, padding_mode=False),
                          ActivationLayer('leakyReLU'),
                          MaxPoolingLayerCUDA((1, 2)),

                          ConvLayer(32, (32, 4), weight_scale=0.088, padding_mode=False),
                          ActivationLayer('leakyReLU'),
                          GlobalPoolingLayer(),

                          FullyConnectedLayer(32, weight_scale=0.125),
                          ActivationLayer('leakyReLU'),
                          FullyConnectedLayer(32, weight_scale=0.125),
                          ActivationLayer('leakyReLU'),
                          FullyConnectedLayer(10, weight_scale=0.17),
                          SoftmaxLayer()],
                         DataProvider(num_genres=10))

    neural_net.init_params_from_file(conv_only=True)

    time1 = time.time()
    neural_net.train(learning_rate=0.005, num_iters=80, lrate_schedule=True)
    time2 = time.time()
    print('Time taken: %.1fs' % (time2 - time1))

    print "\nRESULTS:\n"
    for result in neural_net.results:
        print result
        for val in neural_net.results[result]:
            print val

    neural_net.serialise_params()


def eight_class():
    neural_net = ConvNet([ConvLayer(32, (128, 4), weight_scale=0.044, padding_mode=False),
                          ActivationLayer('leakyReLU'),
                          MaxPoolingLayerCUDA((1, 4)),

                          ConvLayer(32, (32, 4), weight_scale=0.088, padding_mode=False),
                          ActivationLayer('leakyReLU'),
                          MaxPoolingLayerCUDA((1, 2)),

                          ConvLayer(32, (32, 4), weight_scale=0.088, padding_mode=False),
                          ActivationLayer('leakyReLU'),
                          GlobalPoolingLayer(),

                          FullyConnectedLayer(32, weight_scale=0.125),
                          ActivationLayer('leakyReLU'),
                          FullyConnectedLayer(32, weight_scale=0.125),
                          ActivationLayer('leakyReLU'),
                          FullyConnectedLayer(8, weight_scale=0.17),
                          SoftmaxLayer()],
                         DataProvider(num_genres=8))

    neural_net.init_params_from_file(conv_only=True)

    time1 = time.time()
    neural_net.train(learning_rate=0.005, num_iters=80, lrate_schedule=True)
    time2 = time.time()
    print('Time taken: %.1fs' % (time2 - time1))

    print "\nRESULTS:\n"
    for result in neural_net.results:
        print result
        for val in neural_net.results[result]:
            print val

    neural_net.serialise_params()


def six_class(iter_idx):
    neural_net = ConvNet([ConvLayer(32, (128, 4), weight_scale=0.044, padding_mode=False),
                          ActivationLayer('leakyReLU'),
                          MaxPoolingLayerCUDA((1, 4)),

                          ConvLayer(32, (32, 4), weight_scale=0.088, padding_mode=False),
                          ActivationLayer('leakyReLU'),
                          MaxPoolingLayerCUDA((1, 2)),

                          ConvLayer(32, (32, 4), weight_scale=0.088, padding_mode=False),
                          ActivationLayer('leakyReLU'),
                          GlobalPoolingLayer(),

                          FullyConnectedLayer(32, weight_scale=0.125),
                          ActivationLayer('leakyReLU'),
                          FullyConnectedLayer(32, weight_scale=0.125),
                          ActivationLayer('leakyReLU'),
                          FullyConnectedLayer(6, weight_scale=0.17),
                          SoftmaxLayer()],
                         DataProvider(num_genres=6))

    time1 = time.time()
    neural_net.train(learning_rate=0.005, num_iters=80, lrate_schedule=True)
    time2 = time.time()
    print('Time taken: %.1fs' % (time2 - time1))

    f = open('six_class_run' + str(iter_idx) + '.txt', 'w')
    f.write(str(neural_net.results))
    f.close()

    print 'Iteration ' + str(iter_idx)
    print neural_net.results


def four_class(iter_idx):
    neural_net = ConvNet([ConvLayer(32, (128, 4), weight_scale=0.044, padding_mode=False),
                          ActivationLayer('leakyReLU'),
                          MaxPoolingLayerCUDA((1, 4)),

                          ConvLayer(32, (32, 4), weight_scale=0.088, padding_mode=False),
                          ActivationLayer('leakyReLU'),
                          MaxPoolingLayerCUDA((1, 2)),

                          ConvLayer(32, (32, 4), weight_scale=0.088, padding_mode=False),
                          ActivationLayer('leakyReLU'),
                          GlobalPoolingLayer(),

                          FullyConnectedLayer(32, weight_scale=0.125),
                          ActivationLayer('leakyReLU'),
                          FullyConnectedLayer(32, weight_scale=0.125),
                          ActivationLayer('leakyReLU'),
                          FullyConnectedLayer(4, weight_scale=0.17),
                          SoftmaxLayer()],
                         DataProvider(num_genres=4))

    neural_net.setup_layers((128, 599), (4, ))

    time1 = time.time()
    neural_net.train(learning_rate=0.005, num_iters=120, lrate_schedule=True)
    time2 = time.time()
    print('Time taken: %.1fs' % (time2 - time1))

    f = open('four_class_run' + str(iter_idx) + '.txt', 'w')
    f.write(str(neural_net.results))
    f.close()

    print 'Iteration ' + str(iter_idx)
    print neural_net.results


def two_class(iter_idx):
    neural_net = ConvNet([ConvLayer(32, (128, 4), weight_scale=0.044, padding_mode=False),
                          ActivationLayer('leakyReLU'),
                          MaxPoolingLayerCUDA((1, 4)),

                          ConvLayer(32, (32, 4), weight_scale=0.088, padding_mode=False),
                          ActivationLayer('leakyReLU'),
                          MaxPoolingLayerCUDA((1, 2)),

                          ConvLayer(32, (32, 4), weight_scale=0.088, padding_mode=False),
                          ActivationLayer('leakyReLU'),
                          GlobalPoolingLayer(),

                          FullyConnectedLayer(32, weight_scale=0.125),
                          ActivationLayer('leakyReLU'),
                          FullyConnectedLayer(32, weight_scale=0.125),
                          ActivationLayer('leakyReLU'),
                          FullyConnectedLayer(2, weight_scale=0.1),
                          SoftmaxLayer()],
                         DataProvider(num_genres=2))

    time1 = time.time()
    neural_net.train(learning_rate=0.005, num_iters=40, lrate_schedule=True)
    time2 = time.time()
    print('Time taken: %.1fs' % (time2 - time1))

    f = open('two_class_run' + str(iter_idx) + '.txt', 'w')
    f.write(str(neural_net.results))
    f.close()

    print 'Iteration ' + str(iter_idx)
    print neural_net.results


if __name__ == '__main__':
    for i in range(5):
        six_class(i + 26)
