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
                    current_gradient = self.layers[-1].get_gradient(predicted_output,
                                                                    training_example[1])
                    for layer in reversed(self.layers[:-1]):
                        current_gradient = layer.back_prop(current_gradient)

                # After all examples in the current batch have been processed, update parameters for
                # each layer
                for layer in self.layers:
                    layer.update_parameters(learning_rate)

                batch = self.data_provider.get_next_batch()

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

        assert current_shape == cnn_output_shape, "Computed output shape " + current_shape +\
                                                  " does not match given output shape " +\
                                                  cnn_output_shape

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

    def _gradient_check(self, input, output):
        raise NotImplementedError()
