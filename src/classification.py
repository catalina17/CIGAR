from convnet.activation_layer import ActivationLayer
from convnet.conv_layer_cuda import ConvLayerCUDA
from convnet.globalpooling_layer import GlobalPoolingLayer
from convnet.fullyconnected_layer import FullyConnectedLayer
from convnet.maxpooling_layer_cuda import MaxPoolingLayerCUDA
from convnet.softmax_layer import SoftmaxLayer
from convnn import ConvNN
from data_provider import DataProvider

import time


def six_class():
    neural_net = ConvNN([ConvLayerCUDA(64, (128, 4), 0, weight_scale=0.044, padding_mode=False),
                         ActivationLayer('leakyReLU'),
                         MaxPoolingLayerCUDA((1, 4)),

                         ConvLayerCUDA(64, (64, 4), 0, weight_scale=0.0625, padding_mode=False),
                         ActivationLayer('leakyReLU'),
                         MaxPoolingLayerCUDA((1, 2)),

                         ConvLayerCUDA(64, (64, 4), 0, weight_scale=0.0625, padding_mode=False),
                         ActivationLayer('leakyReLU'),
                         MaxPoolingLayerCUDA((1, 2)),

                         ConvLayerCUDA(64, (64, 4), 0, weight_scale=0.0625, padding_mode=False),
                         ActivationLayer('leakyReLU'),
                         GlobalPoolingLayer(),

                         FullyConnectedLayer(64, 0, weight_scale=0.072),
                         ActivationLayer('leakyReLU'),
                         FullyConnectedLayer(32, 0, weight_scale=0.125),
                         ActivationLayer('leakyReLU'),
                         FullyConnectedLayer(6, 0, weight_scale=0.1),
                         SoftmaxLayer()],
                        DataProvider(12, num_genres=6, batch_mode=False))

    neural_net.setup_layers((128, 599), (6, ))

    time1 = time.time()
    neural_net.train(learning_rate=0.01, num_iters=120, lrate_schedule=True)
    time2 = time.time()
    print('Time taken: %.1fs' % (time2 - time1))

    print "\nRESULTS:\n"
    for result in neural_net.results:
        print result
        for val in neural_net.results[result]:
            print val


def four_class():
    neural_net = ConvNN([ConvLayerCUDA(64, (128, 4), 0, weight_scale=0.044, padding_mode=False),
                         ActivationLayer('leakyReLU'),
                         MaxPoolingLayerCUDA((1, 4)),

                         ConvLayerCUDA(64, (64, 4), 0, weight_scale=0.0625, padding_mode=False),
                         ActivationLayer('leakyReLU'),
                         MaxPoolingLayerCUDA((1, 2)),

                         ConvLayerCUDA(64, (64, 4), 0, weight_scale=0.0625, padding_mode=False),
                         ActivationLayer('leakyReLU'),
                         MaxPoolingLayerCUDA((1, 2)),

                         ConvLayerCUDA(64, (64, 4), 0, weight_scale=0.0625, padding_mode=False),
                         ActivationLayer('leakyReLU'),
                         GlobalPoolingLayer(),

                         FullyConnectedLayer(64, 0, weight_scale=0.072),
                         ActivationLayer('leakyReLU'),
                         FullyConnectedLayer(32, 0, weight_scale=0.125),
                         ActivationLayer('leakyReLU'),
                         FullyConnectedLayer(4, 0, weight_scale=0.1),
                         SoftmaxLayer()],
                        DataProvider(8, num_genres=4, batch_mode=False))

    neural_net.setup_layers((128, 599), (4, ))

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


def two_class():
    neural_net = ConvNN([ConvLayerCUDA(64, (128, 4), 0, weight_scale=0.044, padding_mode=False),
                         ActivationLayer('leakyReLU'),
                         MaxPoolingLayerCUDA((1, 4)),

                         ConvLayerCUDA(64, (64, 4), 0, weight_scale=0.0625, padding_mode=False),
                         ActivationLayer('leakyReLU'),
                         MaxPoolingLayerCUDA((1, 2)),

                         ConvLayerCUDA(64, (64, 4), 0, weight_scale=0.0625, padding_mode=False),
                         ActivationLayer('leakyReLU'),
                         GlobalPoolingLayer(),

                         FullyConnectedLayer(64, 0, weight_scale=0.072),
                         ActivationLayer('leakyReLU'),
                         FullyConnectedLayer(32, 0, weight_scale=0.125),
                         ActivationLayer('leakyReLU'),
                         FullyConnectedLayer(2, 0, weight_scale=0.1),
                         SoftmaxLayer()],
                        DataProvider(4, num_genres=2, batch_mode=False))

    time1 = time.time()
    neural_net.train(learning_rate=0.005, num_iters=40, lrate_schedule=True)
    time2 = time.time()
    print('Time taken: %.1fs' % (time2 - time1))

    print "\nRESULTS:\n"
    for result in neural_net.results:
        print result
        for val in neural_net.results[result]:
            print val


if __name__ == '__main__':
    four_class()
