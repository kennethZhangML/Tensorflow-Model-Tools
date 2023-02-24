import tensorflow as tf
from sklearn.model_selection import train_test_split

class OverfitDetectorReducer:
    
    def __init__(self, model, X, y, validation_size=0.2):
        self.model = model
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(X, y, test_size=validation_size)
        self.history = None
        self.best_val_loss = float('inf')
        self.best_model_weights = None
        
    def detect_overfitting(self, epochs=100, batch_size=32, verbose=1):

        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
        history = self.model.fit(self.X_train, self.y_train,
                                 epochs=epochs,
                                 batch_size=batch_size,
                                 validation_data=(self.X_val, self.y_val),
                                 callbacks=[early_stop],
                                 verbose=verbose)

        self.history = history.history
        
        val_loss = self.history['val_loss']

        min_val_loss = min(val_loss)
        min_val_loss_index = val_loss.index(min_val_loss)

        if min_val_loss_index < len(val_loss) - 1:
            return min_val_loss_index + 1
        else:
            return -1
        
    def reduce_overfitting(self, epochs=100, batch_size=32, verbose=1, l1_reg=0.01, l2_reg=0.01, dropout=0.2, optimizer='adam'):
        
        if self.history is None:
            self.detect_overfitting(epochs, batch_size, verbose)
        
        if l1_reg > 0 or l2_reg > 0 or dropout > 0:
            for layer in self.model.layers:
                if isinstance(layer, tf.keras.layers.Dense):
                    if l1_reg > 0:
                        layer.kernel_regularizer = tf.keras.regularizers.l1(l1_reg)
                    if l2_reg > 0:
                        layer.kernel_regularizer = tf.keras.regularizers.l2(l2_reg)
                    if dropout > 0:
                        layer.dropout = dropout
        
        if optimizer == 'adam':
            optimizer = tf.keras.optimizers.Adam()
        
        elif optimizer == 'sgd':
            optimizer = tf.keras.optimizers.SGD()
        
        self.model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        history = self.model.fit(self.X_train, self.y_train,
                                 epochs=epochs,
                                 batch_size=batch_size,
                                 validation_data=(self.X_val, self.y_val),
                                 verbose=verbose)
        
        self.history = history.history
        val_loss = self.history['val_loss']
        min_val_loss = min(val_loss)

        if min_val_loss < self.best_val_loss:
            self.best_val_loss = min_val_loss
            self.best_model_weights = self.model.get_weights()

        self.model.set_weights(self.best_model_weights)
