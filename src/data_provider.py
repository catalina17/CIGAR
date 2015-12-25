import numpy as np
import scipy.misc


class DataProvider(object):

    def __init__(self, batch_size):
        self.batch_size = batch_size

    def get_next_batch(self):
        raise NotImplementedError()

    def get_all_training_data(self):
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()

    def get_input_shape(self):
        raise NotImplementedError()

    def get_output_shape(self):
        raise NotImplementedError()

if __name__ == '__main__':
    im = scipy.misc.imread('test.png')
    a = np.square(im[:,:,:3]/1.0)
    b = np.sqrt(np.sum(a, axis=2)) / 255.0

    c = np.array([0,1])

    te = (b, c)
    print "Input:\n", te[0], "\n", te[0].shape
    print "Output:\n", te[1], "\n", te[1].shape
