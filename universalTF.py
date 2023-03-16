import tensorflow as tf 

import numpy as np
import pandas as pd 

import matplotlib.pyplot as plt

import tensorflow_hub as hub

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers.experimental import preprocessing

import datetime

class UniversalTransferLearningModule(tf.keras.Model):

    def __init__(self, base_dir, base_model_url, image_shape, num_classes):
        super(UniversalTransferLearningModule, self).__init__()

        self.base_model_url = base_model_url # string URL to the internet
        self.image_shape = image_shape # of form (n, n)
        self.num_classes = num_classes # natural number

        if self.num_classes > 2:
            self.activation_func = tf.nn.softmax
            self.class_mode = "binary"
        else:
            self.activation_func = tf.nn.sigmoid
            self.class_mode = "categorical"
        
        self.base_dir = base_dir
        self.train_dir = self.base_dir + "/train"
        self.test_dir = self.test_dir + "/test"

    def build_datasets(self, batch_size):
        self.train_data_generator = ImageDataGenerator(rescale = 1./255)
        self.test_data_generator = ImageDataGenerator(rescale = 1./255)

        self.train_data = self.train_data_generator.flow_from_directory(self.train_dir,
                            batch_size = batch_size, target_size = self.image_shape, label_mode = self.class_mode)
        self.test_data = self.test_data_generator.flow_from_directory(self.test_dir,
                            batch_size = batch_size, target_size = self.image_shape, label_mode = self.class_mode)

    def create_model(self, feature_layer_name):
        '''
        Creates model from KerasLayer given base model url.
        Requires: feature_layer_name is of type Str
        '''
        self.feature_extractor_layer = hub.KerasLayer(self.base_model_url, trainable = False,
                                        name = feature_layer_name, input_shape = self.image_shape + (3,))
        self.model = tf.keras.Sequential([
            self.feature_extractor_layer, 
            tf.keras.layers.Dense(self.num_classes, activation = self.activation_func, 
                                  name = 'output_layer')
        ])

        return self.model

    def create_tensorboard_callback(self, dir_name, experiment_name):
        '''
        Consumes 2 new parameters, dirname and experiment_name:
        - creates a new logging directory to track model training changes
        - returns as a callback to TensorBoard
        Requires: dir_name is of type Str, and experiment_name is of type Str
        '''
        log_dir = dir_name + "/" + experiment_name + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir = log_dir
        )

        return tensorboard_callback

    def build_data_aug(self, layer_list, zoom_fact, height_fact, width_fact, rotat_fact):
        '''
        Requires: layer_list -> array [] str
        Supported data augmentations experimental preprocessing layers (and corresponding code strings):
            - RandomFlip ==> 'RF-h' or 'RF-v' where the trailing h and v representing horizontal/vertical augmentations
            - RandomRotation ==> 'RR' where rotat_fact should be substitued with a float greater than 0 and less than 1.0
            - RandomZoom ==> 'RZ' where the parameter zoom_fact specifies the factor at which the image is zoomed into
            - RandomHeight ==> 'RH' where the parameter rotat_fact specifis the factor at which the image is stretech vertically
            - RandomWidth ==> 'RW' where the parameter width_fact specifies the factor at which the image is stretched horizontally
        '''
        data_aug = tf.keras.Sequential()

        for i, layer_name in enumerate(layer_list):
            if layer_name == 'RF-h' or layer_name == "RF-v":
                vars(self)[f'{i}.{layer_name}'] = preprocessing.RandomFlip('vertical')

            if layer_name == 'RR':
                vars(self)[f'{i}.{layer_name}'] = preprocessing.RandomRotation(rotat_fact)
            
            if layer_name == 'RZ':
                vars(self)[f'{i}{layer_name}'] = preprocessing.RandomZoom(zoom_fact)
            
            if layer_name == 'RH':
                vars(self)[f'{i}{layer_name}'] = preprocessing.RandomHeight(height_fact)
        

        

    def create_base_model(self, data_aug_layerList):

        base_model = tf.keras.applications.EfficientnetB0(include_top = False)
        base_model.trainable = False
    
        inputs = tf.keras.layers.Input(shape = self.image_shape + (3,), name = "input_layer")

        data_aug = self.build_data_aug(data_aug_layerList)

        x = data_aug(inputs)
        x = base_model(x)

        x = tf.keras.layers.GlobalAveragePooling2D(name = "global_average_pooling_layer")(x)
        outputs = tf.keras.layers.Dense(self.num_classes, activation = self.activation_func, name = "output_layer")(x)

        self.model_0 = tf.keras.Model(inputs, outputs)

        self.model_0_layerList = []

        for layer in self.model_0.layers:
            self.model_0_layerList.append(layer)
            
        return self.model_0


    

        

