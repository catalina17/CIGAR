class DataProvider(object):

    def __init__(self, batch_size):
        self.batch_size = batch_size

    def get_next_batch(self):
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()

    def get_input_shape(self):
        raise NotImplementedError()

    def get_output_shape(self):
        raise NotImplementedError()
