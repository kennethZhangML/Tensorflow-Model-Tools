import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

class WeightTransferTracker:

    def __init__(self, model):
        self.model = model
        self.num_layers = len(model.layers)
        self.weight_diffs = [None] * (self.num_layers - 1)
    
    def track_weight_transfers(self, x_train):

        for i in range(self.num_layers - 1):
            weights_before = self.model.layers[i].get_weights()
            self.model.predict(x_train)
            weights_after = self.model.layers[i].get_weights()
            weight_diff = [np.subtract(weights_after[j], weights_before[j]) for j in range(len(weights_before))]

            self.weight_diffs[i] = weight_diff
    
    def plot_weight_transfers(self):

        fig, axs = plt.subplots(self.num_layers - 1)
        
        for i in range(self.num_layers - 1):
            axs[i].imshow(self.weight_diffs[i][0], cmap='gray')
            axs[i].set_title(f"Weight Transfer in Layer {i+1}")
        plt.show()

model = keras.Sequential([
    keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(units=10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = np.expand_dims(x_train, axis=-1) / 255.0
y_train = keras.utils.to_categorical(y_train, num_classes=10)

tracker = WeightTransferTracker(model)

tracker.track_weight_transfers(x_train[:1])
tracker.plot_weight_transfers()
