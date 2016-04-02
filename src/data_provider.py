import numpy as np

import pylab
from scipy.misc import imread


class DataProvider(object):

    def __init__(self, num_genres, genre_dataset_size=100):
        """

        Parameters
        ----------
        num_genres : int
            The number of genres which will be processed by this instance.
        genre_dataset_size : int
            The size of the per-genre dataset.

        """
        self._genre_dataset_size = genre_dataset_size

        self._num_genres = num_genres
        self._genres = ['classical', 'metal', 'blues', 'disco', 'hiphop', 'reggae', 'country',
                        'pop', 'jazz', 'rock']

        self._current_batch_start_index = 0

        self._train_set = None
        self._test_set = None

        # 90:10 proportion for training and test sets
        self._test_indices = np.empty((self._num_genres, self._genre_dataset_size / 10), dtype=int)
        indices = np.array(range(self._genre_dataset_size))

        for i in range(num_genres):
            # The test set for each genre needs is a random partition
            np.random.shuffle(indices)
            self._test_indices[i] = indices[:self._genre_dataset_size / 10]

    @staticmethod
    def get_input_shape():
        """

        Returns
        -------
        tuple
            The shape of the input in the examples provided by DataProvider.

        """
        return (128, 599)

    def get_output_shape(self):
        """

        Returns
        -------
        tuple
            The shape of the output in the examples provided by DataProvider.

        """
        return (self._num_genres,)

    def setup(self):
        """

        Initialises the training and test sets with data from the _data_provider variable.

        """
        self._train_set = np.empty((self._num_genres, self._genre_dataset_size * 9 / 10),
                                   dtype=dict)
        self._test_set = np.empty((self._num_genres, self._genre_dataset_size / 10), dtype=dict)

        train_count = 0
        for genre in self._genres:
            igenre = self._genres.index(genre)
            if igenre >= self._num_genres:
                break

            test_count = 0
            for i in range(self._genre_dataset_size):
                # If the current index represents an example in the test set
                if i in self._test_indices[igenre]:
                    # Add the example to the test set
                    self._test_set[igenre, test_count] = self._get_next_example(genre, i)
                    test_count += 1
                else:
                    # Otherwise add it to the training set
                    self._train_set[igenre, train_count] = self._get_next_example(genre, i)
                    train_count += 1

            np.random.shuffle(self._train_set[igenre])
            train_count = 0

    def get_next_batch(self):
        """

        Returns
        -------
        array of dict
            An array of training examples, or None, if there are no more examples left to be
            processed.

        """
        # If not all training examples have been sent in the previous batches
        if self._current_batch_start_index < self._genre_dataset_size * 9 / 10:
            # Create a new batch
            batch = np.empty(self._num_genres * 2, dtype=dict)
            subbatch_size = 2

            for i in range(self._num_genres):
                batch[subbatch_size * i:subbatch_size * (i + 1)] = \
                    self._train_set[i, self._current_batch_start_index:
                                       self._current_batch_start_index + subbatch_size]

            self._current_batch_start_index += subbatch_size

            np.random.shuffle(batch)
            return batch
        else:
            return None

    def get_all_training_data(self):
        """

        Returns
        -------
        array of dict
            An array containing all training examples.

        """
        return self._train_set.flatten()

    def get_test_data(self):
        """

        Returns
        -------
        array of dict
            An array containing all test examples.

        """
        return self._test_set.flatten()

    def get_test_data_for_genre(self, genre):
        """

        Parameters
        ----------
        genre : int
            The index of the genre we wish to retrieve test data for.

        Returns
        -------
        array of dict
            An array containing all test examples for the given genre index.

        """
        return self._test_set[self._genres.index(genre), :].flatten()

    def reset(self):
        """

        Prepares the instance for a new training iteration.

        """
        self._current_batch_start_index = 0

        # Shuffle training examples as required by stochastic gradient descent
        for i in range(self._num_genres):
            np.random.shuffle(self._train_set[i, :])

    def _get_next_example(self, genre, id):
        """

        Parameters
        ----------
        genre : str
            The name of the genre we wish to retrieve the next example for.
        id : int
            The index of the audio file for which we wish to obtain the corresponding data.

        Returns
        -------
        dict{
            'spec' -> array of float (the grescale intensities of the spectrogram image),
            'out' -> array of float (the encoded correct output of the network for this example;
                                     output[index(genre)] = 1 and output[i] = 0 otherwise),
            'id' -> the index of the audio file
        }

        """
        id_str = '000' + str(id)
        if id < 10:
            id_str = '0' + id_str
        # Create path to spectrogram file
        filename = '../../spectrograms/' + genre + '/' + genre + '.' + id_str + '.png'
        # Read image in array
        im = imread(filename)
        # Convert RGB information to grayscale values
        im_gray = im[:, :, 0] * 0.299 + im[:, :, 1] * 0.587 + im[:, :, 2] * 0.114
        # Normalise
        im_gray /= 255.0

        # Use one-hot encoding to specify the corresponding genre
        output = np.zeros(self._num_genres)
        output[self._genres.index(genre)] = 1

        return dict(spec=im_gray, out=output, id=id)
