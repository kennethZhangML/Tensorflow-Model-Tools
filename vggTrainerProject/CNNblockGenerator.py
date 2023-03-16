import tensorflow as tf 
from tensorflow.keras.layers import Conv2D, Dropout, Flatten, MaxPool2D

class BlockGenerator(tf.keras.Model):

    def __init__(self, filters, kernel_size, repetitions, padding = 'same'):
        super(BlockGenerator, self).__init__()

        self.filters = filters
        self.kernel_size = kernel_size
        self.repetitions = repetitions
        self.padding = padding

        for i in range(self.repetitions):

            vars(self)[f'conv2d_{i}'] = tf.keras.layers.Conv2D(self.filters, self.kernel_size,
                                activation = tf.nn.relu, padding = self.padding)
        
        self.max_pool = tf.keras.layers.MaxPool2D(pool_size = (2, 2))
    
    def call(self, inputs):

        conv2d_0 = vars(self)['conv2d_0']

        x = conv2d_0(inputs)

        for i in range(1, self.repetitions):
            conv2d_i = vars(self)['conv2d_{i}']

            x = conv2d_i(x)
        
        x = self.max_pool(x)
        return x