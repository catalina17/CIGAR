import numpy
from numpy import testing
import unittest

from data_provider import DataProvider


class TestDataProvider(unittest.TestCase):

    def setUp(self):
        self.data_provider = DataProvider(5, genre_dataset_size=40)

    def test_get_output_shape(self):
        self.assertEqual(self.data_provider.get_output_shape(), (5, ))

    def test_setup(self):
        self.data_provider.setup()

        training_data = self.data_provider.get_all_training_data()
        test_data = self.data_provider.get_test_data()
        # Check proportions of training and test sets
        self.assertEqual(training_data.shape[0], 180)
        self.assertEqual(test_data.shape[0], 20)

        ids = numpy.array([])
        for example in training_data:
            ids = numpy.append(ids, example['id'])
        for example in test_data:
            ids = numpy.append(ids, example['id'])
        # Check the reunion of training and test examples gives the entire dataset
        numpy.testing.assert_array_equal(numpy.sort(ids), numpy.sort(numpy.array(5 * (range(40)))))

    def test_get_next_batch(self):
        self.data_provider.setup()

        genres_count = numpy.zeros((5, ), dtype=int)
        for i in range(18):
            batch = self.data_provider.get_next_batch()
            for example in batch:
                genres_count[numpy.argmax(example['out'])] += 1
        numpy.testing.assert_array_equal(genres_count, numpy.array([36, 36, 36, 36, 36]))

    def test_get_all_training_data(self):
        self.data_provider.setup()

        training_data = self.data_provider.get_all_training_data()
        ids = numpy.array([])
        for example in training_data:
            # Check training examples have ids from the dataset range
            self.assertIn(example['id'], range(40))
            ids = numpy.append(ids, example['id'])

    def test_get_test_data(self):
        self.data_provider.setup()

        test_data = self.data_provider.get_test_data()
        ids = numpy.array([])
        for example in test_data:
            # Check test examples have ids from the dataset range
            self.assertIn(example['id'], range(40))
            ids = numpy.append(ids, example['id'])

    def test_get_test_data_for_genre(self):
        self.data_provider.setup()

        test_data_genre = self.data_provider.get_test_data_for_genre('classical')
        ids = numpy.array([])
        for example in test_data_genre:
            ids = numpy.append(ids, example['id'])
        # Check test set for 1 genre does not contain duplicates
        numpy.testing.assert_array_equal(numpy.unique(ids), ids)
        # Check size of test set per genre
        self.assertEqual(ids.shape[0], 4)

    def test_reset(self):
        self.data_provider.setup()

        for i in range(18):
            batch = self.data_provider.get_next_batch()
            self.assertIsNotNone(batch)

        batch = self.data_provider.get_next_batch()
        self.assertIsNone(batch)

        self.data_provider.reset()

        for i in range(18):
            batch = self.data_provider.get_next_batch()
            self.assertIsNotNone(batch)

if __name__ == '__main__':
    TestDataProvider.run()
