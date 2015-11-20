class DataProvider(object):

    def __init__(self, batch_size):
        raise NotImplementedError()

    def get_next_batch(self):
        raise NotImplementedError()
