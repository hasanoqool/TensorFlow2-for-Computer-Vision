"""
CONTACT:
Hasan Al-Oqool : https://www.linkedin.com/in/hasanoqool/
"""

#===============Import the required libraries===============
from PIL import Image
from keras.models import Model, Sequential
from keras.layers import * 
from keras.utils import plot_model
#===============Import the required libraries===============



#===============define a function of model's architecture===============
def model_arch():
    model = Sequential()
    #block1  
    model.add(Input(shape=(64, 64, 3), name='input_layer'))
    model.add(Conv2D(kernel_size=(2, 2), padding='same', strides=(2, 2), filters=32, activation="relu", name='convolution_1'))
    model.add(BatchNormalization(name='batchnormalization_1'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding='same', name='pooling_1'))
    #block2
    model.add(Conv2D(kernel_size=(2, 2), padding='same', strides=(2, 2), filters=64, activation="relu", name='convolution_2'))
    model.add(BatchNormalization(name='batchnormalization_2'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding='same', name='pooling_2'))
    #block3
    model.add(Flatten(name='flatten'))
    model.add(Dense(units=256, activation="relu", name='dense_1'))
    model.add(Dense(units=128, activation="relu", name='dense_2'))
    model.add(Dense(units=3, activation="softmax", name='output'))
    return model
#===============define a function of model's architecture===============

#model summary
summary = model_arch().summary()

#model plot
model = model_arch()
plot_model(model,show_shapes=True,show_layer_names=True,to_file='model_arch.jpg')
# model_diagram = Image.open('/images/model_arch.jpg')