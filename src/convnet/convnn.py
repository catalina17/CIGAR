class ConvNN(object):

    def __init__(self, layers):
        raise NotImplementedError()

    def train(self, input, output, learning_rate, num_iters, batch_size):
        raise NotImplementedError()

    def predict(self, input):
        raise NotImplementedError()

    def _gradient_check(self, input, output):
        raise NotImplementedError()
