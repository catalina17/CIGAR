import numpy as np

from convnet.conv_layer import ConvLayer
from convnet.fullyconnected_layer import FullyConnectedLayer


class ConvNN(object):

    def __init__(self, layers, data_provider):
        """

        Parameters
        ----------
        layers : numpy.array of Layer objects
        data_provider : DataProvider

        """
        self._layers = layers
        self._data_provider = data_provider
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
        for layer in self._layers:
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
        self.results = dict(test=0.0, train=0.0, test_loss=0.0, train_loss=0.0,
                            conf_matrix=np.zeros((self._data_provider.get_output_shape()[0],
                                                  self._data_provider.get_output_shape()[0])))

        # Initialise layers with corresponding input/output dimensions
        self.setup_layers(self._data_provider.get_input_shape(),
                          self._data_provider.get_output_shape())
        self._data_provider.setup()

        for it in range(num_iters):
            print "ConvNN training: iteration #" + str(it + 1)

            self._data_provider.reset()
            batch = self._data_provider.get_next_batch()

            # If available, use the next batch of training examples to train network
            while not(batch is None):
                for training_example in batch:
                    # Forward propagation phase -- calculate output for training example
                    current_input = training_example['spec']
                    for layer in self._layers:
                        current_input = layer.forward_prop(current_input)

                    # Backpropagation phase
                    predicted_output = current_input

                    current_gradient = self._layers[-1].initial_gradient(predicted_output,
                                                                         training_example['out'])
                    for layer in reversed(self._layers[:-1]):
                        current_gradient = layer.back_prop(current_gradient)

                    # Update parameters - online mode
                    for layer in self._layers:
                        if type(layer) in [ConvLayer, FullyConnectedLayer]:
                            if lrate_schedule:
                                layer.update_parameters(learning_rate * (num_iters - it + 1.0) /
                                                        num_iters)
                            else:
                                layer.update_parameters(learning_rate)

                batch = self._data_provider.get_next_batch()

        self._record_training_stats()
        self._record_test_stats()

    def _record_training_stats(self):
        batch = self._data_provider.get_all_training_data()
        error = 0.0
        loss = 0.0

        for training_example in batch:
            current_input = training_example['spec']
            for layer in self._layers:
                current_input = layer.forward_prop(current_input)

            # print "Predicted output: ", current_input, "- True output: ", training_example['out']

            if np.argmax(current_input) != np.argmax(training_example['out']):
                error += 1.0
            loss += self._layers[-1].loss(current_input, training_example['out'])

        print "\nTraining error:\n", error / batch.shape[0]
        print "\nTraining loss:\n", loss / batch.shape[0]

        self.results['train'] = error / batch.shape[0]
        self.results['train_loss'] = loss / batch.shape[0]

    def _record_test_stats(self):
        test_data = self._data_provider.get_test_data()
        test_error = 0.0
        test_loss = 0.0

        for test_example in test_data:
            output = self.predict(test_example['spec'])
            print "Actual ", str(np.argmax(test_example['out']))
            self.results['conf_matrix'][np.argmax(test_example['out'])][np.argmax(output)] += 1

            test_loss += -np.sum(test_example['out'] * np.log(output / np.sum(output)))
            if np.argmax(output) != np.argmax(test_example['out']):
                test_error += 1.0

        print "Test error:", test_error / test_data.shape[0]
        print "Test loss:", test_loss / test_data.shape[0]

        self.results['test'] = test_error / test_data.shape[0]
        self.results['test_loss'] = test_loss / test_data.shape[0]

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
        for layer in self._layers:
            current_input = layer.forward_prop(current_input)

        # Compute predicted output
        print "Predicted ", current_input
        return current_input

    def serialise_params(self):
        count = 0

        for layer in self._layers:
            if type(layer) in [ConvLayer, FullyConnectedLayer]:
                count += 1
                layer.serialise_parameters(count)

    def init_params_from_file(self, conv_only=False):
        count = 0

        for layer in self._layers:
            if conv_only:
                if type(layer) == ConvLayer:
                    count += 1
                    layer.init_parameters_from_file(count)
                    print 'Weights initialised in layer Conv' + str(count)
            else:
                if type(layer) in [ConvLayer, FullyConnectedLayer]:
                    count += 1
                    layer.init_parameters_from_file(count)

    def test_data_activations_for_conv_layer(self, layer_id, genre):
        self._data_provider.setup()
        test_data = self._data_provider.get_test_data_for_genre(genre)
        activations_for_test_data = np.empty(test_data.shape, dtype=dict)

        example_count = -1
        for example in test_data:
            count = 0
            example_count += 1

            current_input = example['spec']
            for layer in self._layers:
                current_input = layer.forward_prop(current_input)
                if type(layer) == ConvLayer:
                    count += 1

                if count == layer_id:
                    break

            activations_for_test_data[example_count] = {
                'filter_activations': current_input,
                'class_prob': self.predict(example['spec'])[np.argmax(example['out'])],
                'id': example['id'],
            }

        return activations_for_test_data
