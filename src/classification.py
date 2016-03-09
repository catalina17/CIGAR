import time

from convnet.activation_layer import ActivationLayer
from convnet.conv_layer import ConvLayer
from convnet.globalpooling_layer import GlobalPoolingLayer
from convnet.fullyconnected_layer import FullyConnectedLayer
from convnet.maxpooling_layer_cuda import MaxPoolingLayerCUDA
from convnet.softmax_layer import SoftmaxLayer
from convnn import ConvNN
from data_provider import DataProvider


def ten_class():
    neural_net = ConvNN([ConvLayer(32, (128, 4), weight_scale=0.044, padding_mode=False),
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
                        DataProvider(20, num_genres=10, batch_mode=False))

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
    neural_net = ConvNN([ConvLayer(32, (128, 4), weight_scale=0.044, padding_mode=False),
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
                        DataProvider(16, num_genres=8, batch_mode=False))

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


def six_class():
    neural_net = ConvNN([ConvLayer(32, (128, 4), weight_scale=0.044, padding_mode=False),
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
                        DataProvider(12, num_genres=6, batch_mode=False))

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


def four_class(iter_idx):
    neural_net = ConvNN([ConvLayer(32, (128, 4), weight_scale=0.044, padding_mode=False),
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
                        DataProvider(8, num_genres=4, batch_mode=False))

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
    neural_net = ConvNN([ConvLayer(32, (128, 4), weight_scale=0.044, padding_mode=False),
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
                        DataProvider(4, num_genres=2, batch_mode=False))

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
    for i in range(14):
        four_class(i + 17)
