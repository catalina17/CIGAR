import numpy as np

from convnet.conv_layer import ConvLayer
from convnet.conv_layer_cuda import ConvLayerCUDA
from convnet.fullyconnected_layer import FullyConnectedLayer
from convnet.fullyconnected_layer_cuda import FullyConnectedLayerCUDA


class ConvNN(object):

    def __init__(self, layers, data_provider):
        """

        Parameters
        ----------
        layers : array of Layer objects
            A sequence of layers representing the architecture of the neural network.
        data_provider : DataProvider

        """
        self._layers = layers
        self._data_provider = data_provider
        self.results = None

    def setup_layers(self, cnn_input_shape, cnn_output_shape):
        """

        Sets the input shapes of all layers in order, checking that the input and output shapes are
        consistent with the architecture of the network.

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

        Performs training of the neural network and saves the training and test statistics
        afterwards.

        Parameters
        ----------
        learning_rate : float
            The (initial) learning rate for updating the parameters.
        num_iters : int
            The number of training iterations.
        lrate_schedule : bool
            Whether a learning schedule is used. The implemented learning schedule is:
                learning_rate(k) = (1 - (k-1)/num_iters) * learning_rate(0).

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
        """

        Saves the following statistics for the training set used:
            - training error (% of examples incorrectly classified)
            - training loss (average cross-entropy loss function for all training examples)

        """
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
        """

        Saves the following statistics for the test set:
            - test error / accuracy (% of examples incorrectly classified)
            - test loss (average cross-entropy loss function for all test examples)

        """
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

        Processes the input and returns the predicted class.

        Parameters
        ----------
        input : numpy.array
            A spectrogram of the audio file we wish to classify.

        Returns
        -------
        array of double
            An array containing the probabilities of the input belonging to each genre, in the
            following order:
                0 - 'classical'
                1 - 'metal'
                2 - 'blues'
                3 - 'disco'
                4 - 'hiphop'
                5 - 'reggae'
                6 - 'country'
                7 - 'pop'
                8 - 'jazz'
                9 - 'rock'

        """
        # Forward propagation
        current_input = input
        for layer in self._layers:
            current_input = layer.forward_prop(current_input)

        # Compute predicted output
        print "Predicted ", current_input
        return current_input

    def serialise_params(self):
        """

        Saves the parameters for convolutional and fully-connected layers to files.

        """
        count = 0

        for layer in self._layers:
            if type(layer) in [ConvLayer, ConvLayerCUDA, FullyConnectedLayer,
                               FullyConnectedLayerCUDA]:
                count += 1
                layer.serialise_parameters(count)

    def init_params_from_file(self, conv_only=False):
        """

        Initialises the neural network with previously saved parameters from the
        corresponding files.

        Parameters
        ----------
        conv_only : bool
            Whether the initialisation is made only for convolutional layers.

        """
        count = 0

        for layer in self._layers:
            if conv_only:
                if type(layer) in [ConvLayer, ConvLayerCUDA]:
                    count += 1
                    layer.init_parameters_from_file(count)
                    print 'Weights initialised in layer Conv' + str(count)
            else:
                if type(layer) in [ConvLayer, ConvLayerCUDA, FullyConnectedLayer,
                                   FullyConnectedLayerCUDA]:
                    count += 1
                    layer.init_parameters_from_file(count)

    def test_data_activations_for_conv_layer(self, layer_id, genre):
        """

        Produces the activations of a specific convolutional layer and a given genre, on the test
        subset corresponding to the genre.

        Parameters
        ----------
        layer_id : int
            The index of the convolutional layer we wish to produce activations for.
        genre : int
            The index of the genre we wish to produce activations for.

        Returns
        -------
        array of dict{
            'filter_activations' -> numpy.array (the activation for the current example),
            'class_prob' -> array of double (the probabilities of the input belonging to each
                                             genre),
            'id' -> the index of the test example being evaluated
        }

        """
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
