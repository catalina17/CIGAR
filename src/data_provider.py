import numpy as np

import pylab
from scipy.misc import imread


class DataProvider(object):

    def __init__(self, batch_size, num_genres, genre_dataset_size=100, batch_mode=False):
        """

        Parameters
        ----------
        batch_size : int
        num_genres : int
        genre_dataset_size : int
        batch_mode : bool

        """
        self._batch_size = batch_size
        self._genre_dataset_size = genre_dataset_size
        self._batch_mode = batch_mode

        self._num_genres = num_genres
        self._genres = ['classical', 'metal', 'blues', 'disco', 'hiphop', 'reggae', 'country',
                        'pop', 'jazz', 'rock']

        self._current_batch_start_index = 0

        self._train_set = None
        self._test_set = None

        self._test_indices = np.empty((self._num_genres, self._genre_dataset_size / 10), dtype=int)
        indices = np.array(range(100))

        np.random.shuffle(indices)
        self._test_indices[0] = indices[:10]
        np.random.shuffle(indices)
        self._test_indices[1] = indices[:10]
        if num_genres > 2:
            np.random.shuffle(indices)
            self._test_indices[2] = indices[:10]
            np.random.shuffle(indices)
            self._test_indices[3] = indices[:10]
        if num_genres > 4:
            np.random.shuffle(indices)
            self._test_indices[4] = indices[:10]
            np.random.shuffle(indices)
            self._test_indices[5] = indices[:10]
        if num_genres > 6:
            np.random.shuffle(indices)
            self._test_indices[6] = indices[:10]
            np.random.shuffle(indices)
            self._test_indices[7] = indices[:10]
        if num_genres > 8:
            np.random.shuffle(indices)
            self._test_indices[8] = indices[:10]
            np.random.shuffle(indices)
            self._test_indices[9] = indices[:10]

    def get_input_shape(self):
        return (128, 599)

    def get_output_shape(self):
        return (self._num_genres,)

    def setup(self):
        if self._batch_mode:
            self._train_set = np.empty(self._genre_dataset_size * 18 / 10, dtype=dict)
        else:
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
                if i in self._test_indices[igenre]:
                    self._test_set[igenre, test_count] = self._get_next_example(genre, i)
                    test_count += 1
                else:
                    if self._batch_mode:
                        self._train_set[train_count] = self._get_next_example(genre, i)
                    else:
                        self._train_set[igenre, train_count] = self._get_next_example(genre, i)
                    train_count += 1

            if not self._batch_mode:
                np.random.shuffle(self._train_set[igenre])
                train_count = 0

        if self._batch_mode:
            np.random.shuffle(self._train_set)

    def get_next_batch(self):
        if self._batch_mode:
            if self._current_batch_start_index < self._genre_dataset_size * 18 / 10:
                batch = self._train_set[self._current_batch_start_index:
                                       self._current_batch_start_index + self._batch_size]
                self._current_batch_start_index += self._batch_size
                return batch
            else:
                return None

        if self._current_batch_start_index < self._genre_dataset_size * 9 / 10:
            batch = np.empty(self._batch_size, dtype=dict)
            subbatch_size = self._batch_size / self._num_genres

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
        return self._train_set.flatten()

    def get_test_data(self):
        return self._test_set.flatten()

    def get_test_data_for_genre(self, genre):
        return self._test_set[self._genres.index(genre), :].flatten()

    def reset(self):
        self._current_batch_start_index = 0

        if self._batch_mode:
            np.random.shuffle(self._train_set)
        else:
            for i in range(self._num_genres):
                np.random.shuffle(self._train_set[i, :])

    def _get_next_example(self, genre, id):
        """

        Parameters
        ----------
        genre : str
        id : int

        Returns
        -------
        tuple

        """
        id_str = '000' + str(id)
        if id < 10:
            id_str = '0' + id_str
        filename = '../../spectrograms/' + genre + '/' + genre + '.' + id_str + '.png'
        im = imread(filename)
        im_gray = im[:, :, 0] * 0.299 + im[:, :, 1] * 0.587 + im[:, :, 2] * 0.114
        im_gray /= 255.0

        output = np.zeros(self._num_genres)
        output[self._genres.index(genre)] = 1

        return dict(spec=im_gray, out=output, id=id)

if __name__ == '__main__':
    data_provider = DataProvider(10, num_genres=4, batch_mode=False)
    data_provider.setup()

    a = data_provider.get_all_training_data()
    b = data_provider.get_test_data()

    print "TRAINING DATA"
    print a
    print "TEST DATA"
    print b
