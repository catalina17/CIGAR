import numpy as np
import pylab
from scipy.misc import imread


class DataProvider(object):

    def __init__(self, batch_size, num_genres=2, genre_dataset_size=100):
        """

        Parameters
        ----------
        batch_size : int
        num_genres : int
        genre_dataset_size : int

        """
        self.batch_size = batch_size
        self.genre_dataset_size = genre_dataset_size

        self.num_genres = num_genres
        self.genres = ['classical', 'metal']

        self.current_batch_start_index = 0

        self.dataset = None

    def get_input_shape(self):
        return (128, 599)

    def get_output_shape(self):
        return (2,)

    def setup(self):
        self.dataset = np.empty((self.num_genres, self.genre_dataset_size), dtype=dict)
        for genre in self.genres:
            igenre = 0
            if genre == 'metal':
                igenre = 1

            for i in range(self.genre_dataset_size):
                self.dataset[igenre, i] = self._get_next_example(genre, i)

            np.random.shuffle(self.dataset[igenre])

        print "Test set - CLASSICAL:"
        for ex in self.dataset[0, self.genre_dataset_size * 8 / 10:]:
            print ex['id']
        print "Test set - METAL:"
        for ex in self.dataset[1, self.genre_dataset_size * 8 / 10:]:
            print ex['id']

    def get_next_batch(self):
        if self.current_batch_start_index < self.genre_dataset_size * 8 / 10:
            batch = np.empty(self.batch_size, dtype=dict)
            subbatch_size = self.batch_size / self.num_genres

            for i in range(self.num_genres):
                batch[subbatch_size * i:subbatch_size * (i + 1)] =\
                    self.dataset[i, self.current_batch_start_index:
                                    self.current_batch_start_index + subbatch_size]

            self.current_batch_start_index += subbatch_size

            np.random.shuffle(batch)
            return batch
        else:
            return None

    def get_all_training_data(self):
        return self.dataset[:, :self.genre_dataset_size * 8 / 10].flatten()

    def get_test_data(self):
        return self.dataset[:, self.genre_dataset_size * 8 / 10:].flatten()

    def reset(self):
        self.current_batch_start_index = 0
        for i in range(self.num_genres):
            np.random.shuffle(self.dataset[i, :self.genre_dataset_size * 8 / 10])

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
        if genre == 'classical':
            output[0] = 1
        else:
            output[1] = 1

        return dict(spec=im_gray, out=output, id=id)

if __name__ == '__main__':
    data_provider = DataProvider(10)
    data_provider.setup()

    a = data_provider.get_all_training_data()
    b = data_provider.get_test_data()

    print a.shape
    print b.shape
