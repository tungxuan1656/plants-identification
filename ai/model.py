from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model
from keras.layers import Input, Dense, Activation, Lambda, Flatten
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras.optimizers import Adam
import numpy as np
import PIL
import os
from tqdm.notebook import tqdm
import shutil


train_datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.25, height_shift_range=0.25, horizontal_flip=True, preprocessing_function=preprocess_input)
img_size = (256, 256)
train_generator = train_datagen.flow_from_directory(
    './dataset/plants',
    target_size=img_size,
    batch_size=2,
    class_mode='categorical',
    shuffle=False)

a, b = train_generator.next()
print(b)


# model v1
# model = VGG16(include_top=False, input_shape=(256, 256, 3))
# for layer in model.layers:
#     layer.trainable=False
# flat1 = Flatten()(model.layers[-1].output)
# class1 = Dense(512, activation='relu')(flat1)
# # class1 = Dense(256, activation='relu')(class1)
# output = Dense(10, activation='softmax')(class1)
# # define new model
# model = Model(inputs=model.inputs, outputs=output)
