from convnn import ConvNN
from convnet.conv_layer import ConvLayer
from convnet.activation_layer import ActivationLayer
from convnet.maxpooling_layer_cuda import MaxPoolingLayerCUDA
from convnet.globalpooling_layer import GlobalPoolingLayer
from convnet.fullyconnected_layer import FullyConnectedLayer
from convnet.softmax_layer import SoftmaxLayer
from data_provider import DataProvider

import matplotlib.pyplot as plt
import numpy as np
import os

if __name__ == '__main__':
    neural_net = ConvNN([ConvLayer(32, (128, 4), 0, weight_scale=0.044, padding_mode=False),
                         ActivationLayer('leakyReLU'),
                         MaxPoolingLayerCUDA((1, 4)),

                         ConvLayer(32, (32, 4), 0, weight_scale=0.088, padding_mode=False),
                         ActivationLayer('leakyReLU'),
                         MaxPoolingLayerCUDA((1, 2)),

                         ConvLayer(32, (32, 4), 0, weight_scale=0.088, padding_mode=False),
                         ActivationLayer('leakyReLU'),
                         GlobalPoolingLayer(),

                         FullyConnectedLayer(32, 0, weight_scale=0.125),
                         ActivationLayer('leakyReLU'),
                         FullyConnectedLayer(32, 0, weight_scale=0.125),
                         ActivationLayer('leakyReLU'),
                         FullyConnectedLayer(4, 0, weight_scale=0.17),
                         SoftmaxLayer()],
                        DataProvider(8, num_genres=4, batch_mode=False))

    neural_net.setup_layers((128, 599), (4, ))
    neural_net.init_params_from_file()

    for layer_id in range(3):
        activations_for_test_data = neural_net.test_data_activations_for_conv_layer(layer_id + 1,
                                                                                    'blues')

        for result in activations_for_test_data:
            print result['filter_activations'].shape[1]
            for filter_idx in range(result['filter_activations'].shape[0]):
                x = np.arange(0, result['filter_activations'].shape[1], 1)
                y = result['filter_activations'][filter_idx]

                dir_path = './activations/conv_layer' + str(layer_id + 1) + '/filter' +\
                           str(filter_idx + 1)
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path)

                plt.plot(x, y)
                plt.gcf().set_size_inches(7.72903226, 1.6)
                plt.axis('off')
                plt.gca().set_axis_off()
                plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
                plt.margins(0, 0)
                plt.gca().xaxis.set_major_locator(plt.NullLocator())
                plt.gca().yaxis.set_major_locator(plt.NullLocator())

                plt.savefig(dir_path + '/blues' + str(result['id']) + '_p=' +
                            str(round(result['class_prob'], 2)) + '.png',
                            bbox_inches="tight", pad_inches=0)
                plt.clf()
