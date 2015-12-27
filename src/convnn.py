from convnet.activation_layer import ActivationLayer
from convnet.conv_layer import ConvLayer
from convnet.globalpooling_layer import GlobalPoolingLayer
from convnet.fullyconnected_layer import FullyConnectedLayer
from convnet.maxpooling_layer import MaxPoolingLayer
from convnet.softmax_layer import SoftmaxLayer
from data_provider import DataProvider

import numpy as np
import time


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
            # print "DONE setting up " + str(type(layer)) + '\n'

        assert current_shape == cnn_output_shape, "Computed output shape " + str(current_shape) +\
                                                  " does not match given output shape " +\
                                                  str(cnn_output_shape)

        # print "ConvNN setup successful!"

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
        self.data_provider.setup()

        for it in range(num_iters):
            print "ConvNN training: iteration #" + str(it + 1)

            self.data_provider.reset()
            batch = self.data_provider.get_next_batch()
            batch_count = 0
            # If available, use the next batch of training examples to train network
            while not(batch is None):
                batch_count += 1
                print "Batch #", str(batch_count)

                count = 0
                for training_example in batch:
                    count += 1

                    # Forward propagation phase -- calculate output for training example
                    current_input = training_example['spec']
                    for layer in self.layers:
                        current_input = layer.forward_prop(current_input)

                    # Backpropagation phase
                    predicted_output = current_input
                    print "Predicted output: ", predicted_output,\
                          "--- True output: ", training_example['out']

                    current_gradient = self.layers[-1].initial_gradient(predicted_output,
                                                                        training_example['out'])
                    for layer in reversed(self.layers[:-1]):
                        current_gradient = layer.back_prop(current_gradient)
                        # print np.std(current_gradient)

                    # Update parameters - online mode
                    for layer in self.layers:
                        if type(layer) in [ConvLayer, FullyConnectedLayer]:
                            layer.update_parameters(learning_rate)

                batch = self.data_provider.get_next_batch()

            all_training_data = self.data_provider.get_all_training_data()
            self.error_and_loss(all_training_data)

            self.test()

    def error_and_loss(self, batch):
        error = 0.0
        loss = 0.0

        for training_example in batch:
            current_input = training_example['spec']
            for layer in self.layers:
                current_input = layer.forward_prop(current_input)

            print "Predicted output: ", current_input, "--- True output: ", training_example['out']

            if np.argmax(current_input) != np.argmax(training_example['out']):
                error += 1.0
            loss += self.layers[-1].loss(current_input, training_example['out'])

        print "\nTraining error:\n", error / batch.shape[0]
        print "\nTraining loss:\n", loss

    def error(self, batch):
        error = 0.0

        for training_example in batch:
            current_input = training_example['spec']
            for layer in self.layers:
                current_input = layer.forward_prop(current_input)

            if np.argmax(current_input) != np.argmax(training_example['out']):
                error += 1.0

        print error
        error /= batch.shape[0]
        return error

    def training_loss(self, training_batch):
        loss = 0.0

        for training_example in training_batch:
            current_input = training_example['spec']
            for layer in self.layers:
                current_input = layer.forward_prop(current_input)

            loss += self.layers[-1].loss(current_input, training_example['out'])

        loss /= training_batch.shape[0]
        return loss

    def test(self):
        test_data = self.data_provider.get_test_data()
        test_error = 0.0

        for test_example in test_data:
            output_class = self.predict(test_example['spec'])
            print "Actual ", str(np.argmax(test_example['out']))
            if output_class != np.argmax(test_example['out']):
                test_error += 1.0

        print "Test error:", test_error / test_data.shape[0]

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
        print "Predicted ", current_input
        predicted_class = np.argmax(current_input)
        return predicted_class

if __name__ == '__main__':
    neural_net = ConvNN([ConvLayer(64, (128, 4), 0, 0.044, False),
                         ActivationLayer('ReLU'),
                         MaxPoolingLayer((1, 4)),

                         ConvLayer(64, (64, 4), 0, 0.0625, False),
                         ActivationLayer('ReLU'),
                         MaxPoolingLayer((1, 2)),

                         ConvLayer(64, (64, 4), 0, 0.0625, False),
                         ActivationLayer('ReLU'),
                         GlobalPoolingLayer(),

                         FullyConnectedLayer(64, weight_decay=0, weight_scale=0.072),
                         ActivationLayer('ReLU'),
                         FullyConnectedLayer(32, weight_decay=0, weight_scale=0.125),
                         ActivationLayer('ReLU'),
                         FullyConnectedLayer(2, weight_decay=0, weight_scale=0.1),
                         SoftmaxLayer()],
                        DataProvider(4))

    neural_net._setup_layers((128, 599), (2, ))

    time1 = time.time()
    neural_net.train(learning_rate=0.01, num_iters=20)
    time2 = time.time()
    print('Time taken: %.1fs' % (time2 - time1))
