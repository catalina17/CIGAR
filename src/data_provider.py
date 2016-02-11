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
        self.batch_size = batch_size
        self.genre_dataset_size = genre_dataset_size
        self.batch_mode = batch_mode

        self.num_genres = num_genres
        self.genres = ['classical', 'metal', 'blues', 'disco', 'hiphop', 'reggae']

        self.current_batch_start_index = 0

        self.train_set = None
        self.test_set = None

        self.test_indices = np.empty((self.num_genres, self.genre_dataset_size * 2 / 10), dtype=int)
        self.test_indices[0] = np.array([9,13,2,6,20,11,63,58,23,51,76,99,50,70,85,55,73,91,80,22])
        self.test_indices[1] = np.array([99,42,9,32,0,91,84,80,59,12,4,56,5,27,38,23,19,18,87,69])
        if num_genres > 2:
            self.test_indices[2] = np.array([22,58,55,76,39,56,37,77,3,66,33,78,49,27,32,24,44,8,95,
                                             88])
            self.test_indices[3] = np.array([52,0,84,62,7,18,14,47,70,98,56,90,24,17,42,13,28,45,3,
                                             38])
        if num_genres > 4:
            self.test_indices[4] = np.array([39,24,4,18,79,12,40,56,64,28,52,15,88,83,91,96,81,11,
                                             36,45])
            self.test_indices[5] = np.array([62,78,3,67,51,32,75,35,90,97,58,92,19,44,33,42,87,74,
                                             57,53])

    def get_input_shape(self):
        return (128, 599)

    def get_output_shape(self):
        return (self.num_genres,)

    def setup(self):
        if self.batch_mode:
            self.train_set = np.empty(self.genre_dataset_size * 16 / 10, dtype=dict)
        else:
            self.train_set = np.empty((self.num_genres, self.genre_dataset_size * 8 / 10),
                                      dtype=dict)
        self.test_set = np.empty((self.num_genres, self.genre_dataset_size * 2 / 10), dtype=dict)

        train_count = 0
        for genre in self.genres:
            igenre = self.genres.index(genre)
            if igenre >= self.num_genres:
                break

            test_count = 0
            for i in range(self.genre_dataset_size):
                if i in self.test_indices[igenre]:
                    self.test_set[igenre, test_count] = self._get_next_example(genre, i)
                    test_count += 1
                else:
                    if self.batch_mode:
                        self.train_set[train_count] = self._get_next_example(genre, i)
                    else:
                        self.train_set[igenre, train_count] = self._get_next_example(genre, i)
                    train_count += 1

            if not self.batch_mode:
                np.random.shuffle(self.train_set[igenre])
                train_count = 0

        if self.batch_mode:
            np.random.shuffle(self.train_set)

    def get_next_batch(self):
        if self.batch_mode:
            if self.current_batch_start_index < self.genre_dataset_size * 16 / 10:
                batch = self.train_set[self.current_batch_start_index:
                                       self.current_batch_start_index + self.batch_size]
                self.current_batch_start_index += self.batch_size
                return batch
            else:
                return None

        if self.current_batch_start_index < self.genre_dataset_size * 8 / 10:
            batch = np.empty(self.batch_size, dtype=dict)
            subbatch_size = self.batch_size / self.num_genres

            for i in range(self.num_genres):
                batch[subbatch_size * i:subbatch_size * (i + 1)] =\
                    self.train_set[i, self.current_batch_start_index:
                                      self.current_batch_start_index + subbatch_size]

            self.current_batch_start_index += subbatch_size

            np.random.shuffle(batch)
            return batch
        else:
            return None

    def get_all_training_data(self):
        return self.train_set.flatten()

    def get_test_data(self):
        return self.test_set.flatten()

    def get_test_data_for_genre(self, genre):
        return self.test_set[self.genres.index(genre), :].flatten()

    def reset(self):
        self.current_batch_start_index = 0

        if self.batch_mode:
            np.random.shuffle(self.train_set)
        else:
            for i in range(self.num_genres):
                np.random.shuffle(self.train_set[i, :])

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

        output = np.zeros(self.num_genres)
        output[self.genres.index(genre)] = 1

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
