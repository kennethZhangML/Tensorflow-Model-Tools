import tensorflow as tf 
import matplotlib.pyplot as plt
from matplotlib.image import imread

import seaborn as sns
import numpy as np 
import os 

import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPool2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from vggGenerator import VGGGenerator

class VGGTrainer():

    def __init__(self, base_dir, epochs, color_mode, class_mode, target_image_size):

        self.base_dir = base_dir 
        self.epochs = epochs

        self.target_image_size = target_image_size
        self.color_mode = color_mode
        self.class_mode = class_mode

        self.train_dir = os.path.join(self.base_dir + "/train")
        self.test_dir = os.path.join(self.base_dir + "/test")

    def create_generators(self):
        train_datagenerator = ImageDataGenerator(rescale = 1./255, rotation_range = 20,
                width_shift_range = 0.1, height_shift_range = 0.1, shear_range = 0.1, zoom_range = 0.1, 
                horizontal_flip = True, fill_mode = "nearest")
        test_datagenerator = ImageDataGenerator(rescale = 1./255)

        self.train_data = train_datagenerator.flow_from_directory(directory = self.train_dir, target_size = self.target_image_size, 
                        color_mode = self.color_mode, class_mode = self.class_mode, shuffle = True)
        self.test_data = test_datagenerator.flow_from_directory(directory =  self.test_dir, shuffle = False)

        self.num_classes = self.train_data.num_classes

        return self.train_data, self.test_data
    
    def visualize_dimDistr(self, dir_name, class_name):

        dimension1, dimension2 = [], []

        base_img_path = self.base_dir + "/" + dir_name + "/" + class_name

        for file_name in os.listdir(base_img_path):
            img = imread(base_img_path + "/" + file_name)
            d1, d2, color = img.shape 
            dimension1.append(d1)
            dimension2.append(d2)

        fig = sns.jointplot(dimension1, dimension2)
        fig.set_axis_labels('Dimension 1', 'Dimension 2', fontsize = 16)
        fig.figure.tight_layout()

    def build_vgg(self):

        if self.num_classes == 2:
            self.loss = "binary_crossentropy"
            self.optimizer = "adam"
        else:
            self.loss = "sparse_categorical_crossentropy"
            self.optimizer = "adam"
        
        self.model = VGGGenerator(num_classes = self.num_classes)
        self.model.compile(loss = self.loss, optimizer = self.optimizer)

        for i, layer in enumerate(self.model.layers):
            print(f'{i}. {layer}')
        
    
    def fit_vgg(self):

        try:
            self.model.fit(self.train_data, validation_data = self.test_data)

        except NotImplementedError:
            
            print("Model not implemented, get input and outputs from Functional API.")