from convnet.conv_layer import ConvLayer
from convnet.fullyconnected_layer import FullyConnectedLayer

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
        self.results = None

    def setup_layers(self, cnn_input_shape, cnn_output_shape):
        """

        Parameters
        ----------
        cnn_input_shape : tuple
        cnn_output_shape : tuple

        """
        current_shape = cnn_input_shape
        print "OUTPUT shapes:"
        for layer in self.layers:
            layer.set_input_shape(current_shape)
            current_shape = layer.get_output_shape()
            print layer.__class__.__name__, current_shape

        assert current_shape == cnn_output_shape, "Computed output shape " + str(current_shape) +\
                                                  " does not match given output shape " +\
                                                  str(cnn_output_shape)

        # print "ConvNN setup successful!"

    def train(self, learning_rate, num_iters, lrate_schedule=False):
        """

        Parameters
        ----------
        learning_rate : float
        num_iters : int
        lrate_schedule : bool

        """
        self.results = dict(test=np.zeros(num_iters), train=np.zeros(num_iters),
                            train_loss=np.zeros(num_iters), test_loss=np.zeros(num_iters),
                            mse=np.zeros(num_iters), test_mse=np.zeros(num_iters))

        # Initialise layers with corresponding input/output dimensions
        self.setup_layers(self.data_provider.get_input_shape(),
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
                # print "Batch #", str(batch_count)

                count = 0
                for training_example in batch:
                    count += 1

                    # Forward propagation phase -- calculate output for training example
                    current_input = training_example['spec']
                    for layer in self.layers:
                        current_input = layer.forward_prop(current_input)

                    # Backpropagation phase
                    predicted_output = current_input
                    # print "Predicted: ", predicted_output,\
                    #      "--- Actual: ", training_example['out'], "-", str(training_example['id'])

                    current_gradient = self.layers[-1].initial_gradient(predicted_output,
                                                                        training_example['out'])
                    for layer in reversed(self.layers[:-1]):
                        current_gradient = layer.back_prop(current_gradient)

                    # Update parameters - online mode
                    for layer in self.layers:
                        if type(layer) in [ConvLayer, FullyConnectedLayer]:
                            if lrate_schedule:
                                layer.update_parameters(learning_rate * (num_iters - it + 1.0) /
                                                        num_iters)
                            else:
                                layer.update_parameters(learning_rate)

                batch = self.data_provider.get_next_batch()

            all_training_data = self.data_provider.get_all_training_data()
            self.error_and_loss(all_training_data, it)

            self.test(it)

    def error_and_loss(self, batch, iter_idx):
        error = 0.0
        mse = 0.0
        loss = 0.0

        for training_example in batch:
            current_input = training_example['spec']
            for layer in self.layers:
                current_input = layer.forward_prop(current_input)

            # print "Predicted output: ", current_input, "- True output: ", training_example['out']

            if np.argmax(current_input) != np.argmax(training_example['out']):
                error += 1.0
                mse += np.sum((current_input - training_example['out']) ** 2)
            loss += self.layers[-1].loss(current_input, training_example['out'])

        print "\nTraining error:\n", error / batch.shape[0]
        print "\nTraining MSE:\n", mse / batch.shape[0]
        print "\nTraining loss:\n", loss / batch.shape[0]

        self.results['train'][iter_idx] = error / batch.shape[0]
        self.results['mse'][iter_idx] = mse / batch.shape[0]
        self.results['train_loss'][iter_idx] = loss / batch.shape[0]

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

    def test(self, iter_idx):
        test_data = self.data_provider.get_test_data()
        test_error = 0.0
        test_mse = 0.0
        test_loss = 0.0

        for test_example in test_data:
            output = self.predict(test_example['spec'])
            print "Actual ", str(np.argmax(test_example['out']))

            test_loss += -np.sum(test_example['out'] * np.log(output / np.sum(output)))
            if np.argmax(output) != np.argmax(test_example['out']):
                test_error += 1.0
                test_mse += np.sum((output - test_example['out']) ** 2)

        print "Test error:", test_error / test_data.shape[0]
        print "Test MSE:", test_mse / test_data.shape[0]
        print "Test loss:", test_loss / test_data.shape[0]

        self.results['test'][iter_idx] = test_error / test_data.shape[0]
        self.results['test_mse'][iter_idx] = test_mse / test_data.shape[0]
        self.results['test_loss'][iter_idx] = test_loss / test_data.shape[0]

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
        return current_input

    def serialise_params(self):
        count = 0

        for layer in self.layers:
            if type(layer) in [ConvLayer, FullyConnectedLayer]:
                count += 1
                layer.serialise_parameters(count)

    def init_params_from_file(self):
        count = 0

        for layer in self.layers:
            if type(layer) in [ConvLayer, FullyConnectedLayer]:
                count += 1
                layer.init_parameters_from_file(count)
