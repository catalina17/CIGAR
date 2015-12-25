from convnet.activation_layer import ActivationLayer
from convnet.conv_layer import ConvLayer
from convnet.globalpooling_layer import GlobalPoolingLayer
from convnet.fullyconnected_layer import FullyConnectedLayer
from convnet.maxpooling_layer import MaxPoolingLayer
from convnet.softmax_layer import SoftmaxLayer
from data_provider import DataProvider

import numpy as np


class ConvNN(object):

    def __init__(self, layers, data_provider):
        """

        Parameters
        ----------
        layers : numpy.array of Layer objects
        data_provider : DataProvider

        """
        self.layers = layers
        self.data_provider = data_provider

    def _setup_layers(self, cnn_input_shape, cnn_output_shape):
        """

        Parameters
        ----------
        cnn_input_shape : tuple
        cnn_output_shape : tuple

        """
        current_shape = cnn_input_shape
        for layer in self.layers:
            layer.set_input_shape(current_shape)
            current_shape = layer.get_output_shape()
            print "DONE setting up " + str(type(layer)) + '\n'

        assert current_shape == cnn_output_shape, "Computed output shape " + str(current_shape) +\
                                                  " does not match given output shape " +\
                                                  str(cnn_output_shape)

        print "ConvNN setup successful!"

    def train(self, learning_rate, num_iters):
        """

        Parameters
        ----------
        learning_rate : float
        num_iters : int

        """
        # Initialise layers with corresponding input/output dimensions
        self._setup_layers(self.data_provider.get_input_shape(),
                           self.data_provider.get_output_shape())

        for it in range(num_iters):
            print "ConvNN training: iteration #" + str(it + 1)

            self.data_provider.reset()
            batch = self.data_provider.get_next_batch()
            # If available, use the next batch of training examples to train network
            while batch:
                for training_example in batch:
                    # Forward propagation phase -- calculate output for training example
                    current_input = training_example[0]
                    for layer in self.layers:
                        current_input = layer.forward_prop(current_input)

                    # Backpropagation phase
                    predicted_output = current_input
                    current_gradient = self.layers[-1].initial_gradient(predicted_output,
                                                                        training_example[1])
                    for layer in reversed(self.layers[:-1]):
                        current_gradient = layer.back_prop(current_gradient)

                # After all examples in the current batch have been processed, update parameters
                for layer in self.layers:
                    if type(layer) in [ConvLayer, FullyConnectedLayer]:
                        layer.update_parameters(learning_rate)

                batch = self.data_provider.get_next_batch()

    def error(self, training_batch):
        raise NotImplementedError()

    def training_loss(self, training_batch):
        raise NotImplementedError()

    def predict(self, input):
        """

        Parameters
        ----------
        input : numpy.array

        Returns
        -------
        int

        """
        # Forward propagation
        current_input = input
        for layer in self.layers:
            current_input = layer.forward_prop(current_input)

        # Compute predicted output
        predicted_class = np.argmax(current_input)
        return predicted_class

if __name__ == '__main__':
    neural_net = ConvNN([ConvLayer(256, (128, 4), 0.1, False),
                         ActivationLayer('ReLU'),
                         MaxPoolingLayer((1, 4)),

                         ConvLayer(256, (256, 4), 0.1, False),
                         ActivationLayer('ReLU'),
                         MaxPoolingLayer((1, 2)),

                         ConvLayer(512, (256, 4), 0.1, False),
                         ActivationLayer('ReLU'),
                         MaxPoolingLayer((1, 2)),

                         ConvLayer(512, (512, 4), 0.1, False),
                         ActivationLayer('ReLU'),
                         GlobalPoolingLayer(),

                         FullyConnectedLayer(2048, weight_decay=0.1, weight_scale=0.01),
                         FullyConnectedLayer(2048, weight_decay=0.1, weight_scale=0.01),
                         FullyConnectedLayer(40, weight_decay=0.1, weight_scale=0.01),
                         FullyConnectedLayer(2, weight_decay=0.1, weight_scale=0.01),
                         SoftmaxLayer()],
                        DataProvider(20))

    neural_net._setup_layers((128, 599), (2, ))

    dummy_input = np.random.uniform(0, 255, [128, 599])
    # dummy_input /= np.sum(dummy_input)
    print "CNN input:\n", dummy_input

    for layer in neural_net.layers:
        dummy_input = layer.forward_prop(dummy_input)
        print "LAYER--->>:", str(type(layer))
        print dummy_input

    print "\n--->>BACKPROPAGATION\n"
    dummy_output = np.array([1, 0])
    current_gradient = neural_net.layers[-1].initial_gradient(dummy_input, dummy_output)
    print "\nInitial gradient:\n", current_gradient

    for layer in reversed(neural_net.layers[:-1]):
        current_gradient = layer.back_prop(current_gradient)
        print "LAYER--->>:", str(type(layer))
        print current_gradient
