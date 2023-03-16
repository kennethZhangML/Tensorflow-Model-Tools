import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dropout, Dense, Flatten, MaxPool2D

from CNNblockGenerator import BlockGenerator

class VGGGenerator(tf.keras.Model):

    def __init__(self, num_classes):
        super(VGGGenerator, self).__init__()

        self.num_classes = num_classes

        self.block_1 = BlockGenerator(filters = 64, kernel_size = (3, 3), repetitions = 2)
        self.block_2 = BlockGenerator(filters = 128, kernel_size = (3, 3), repetitions = 2)
        self.block_3 = BlockGenerator(filters = 256, kernel_size = (3, 3), repetitions = 2)
        self.block_4 = BlockGenerator(filters = 512, kernel_size = (3, 3), repetitions = 2)

        self.fc_1 = tf.keras.layers.Dense(512, activation = tf.nn.relu)
        self.fc_2 = tf.keras.layers.Dense(256, activation = tf.nn.relu)
        self.fc_3 = tf.keras.layers.Dense(128, activation = tf.nn.relu)
        self.fc_4 = tf.keras.layers.Dense(64, activation = tf.nn.relu)

        if self.num_classes == 2:
            self.fc_5 = tf.keras.layers.Dense(2, activation = tf.nn.sigmoid)
        else:
            self.fc_5 = tf.keras.layers.Dense(self.num_classes, activation = tf.nn.softmax)

    def forward(self, inputs):
        
        x = self.block_1(inputs)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = self.fc_1(x)
        x = self.fc_2(x)
        x = self.fc_3(x)
        x = self.fc_4(x)

        outputs = self.fc_5(x)
        return outputs