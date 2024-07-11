import tensorflow as tf
from keras import initializers

class ListInitializer(initializers.Initializer):
    def __init__(self, weights_list):
        self.weights = tf.constant(weights_list, shape=(4, 1))


    def __call__(self, shape, dtype=None):
        # Ensure the shape of the weights matches the expected shape
        assert shape == self.weights.shape
        return self.weights
