"""
CONTACT:
Hasan Al-Oqool : https://www.linkedin.com/in/hasanoqool/
"""

#===============Import the required libraries===============
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from keras import Model
from keras.datasets import mnist
from keras.layers import * 
from keras.models import * 
#===============Import the required libraries===============


#===============function will load, normalizing the train and test sets and one-hot encoding the labels===============
def load_data():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    #Normalization the data
    X_train = X_train.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0
    # Reshape grayscale to include 1 channel dimension.
    X_train = np.expand_dims(X_train, axis=3)
    X_test = np.expand_dims(X_test, axis=3)
    # Label encoding (one-hot)
    label_binarizer = LabelBinarizer()
    y_train = label_binarizer.fit_transform(y_train)
    y_test = label_binarizer.fit_transform(y_test)
    return X_train, y_train, X_test, y_test
#===============function will load, normalizing the train and test sets and one-hot encoding the labels===============


#===============function for building network===============
def build_network():
    input_layer = Input(shape=(28, 28, 1))
    convolution_1 = Conv2D(filters=32,kernel_size=(2, 2), padding='same', strides=(2, 2))(input_layer)
    activation_1 = ReLU()(convolution_1)
    batch_normalization_1 = BatchNormalization()(activation_1)
    pooling_1 = MaxPooling2D(pool_size=(2, 2),strides=(1, 1),padding='same')(batch_normalization_1)
    dropout = Dropout(rate=0.5)(pooling_1)
    flatten = Flatten()(dropout)
    dense_1 = Dense(units=128)(flatten)
    activation_2 = ReLU()(dense_1)
    dense_2 = Dense(units=10)(activation_2)
    output = Softmax()(dense_2)
    network = Model(inputs=input_layer, outputs=output)
    return network
#===============function for building network===============


#===============function for evaluating the network===============
def evaluate(model, X_test, y_test):
    _, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f'Accuracy: {accuracy}')
#===============function for evaluating the network===============


#===============Prepare the data, create a validation split===============
X_train, y_train, X_test, y_test = load_data()
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2)
model = build_network()
#===============Prepare the data, create a validation split===============


#===============Compile and train the model for 25 epochs, with a batch size of 1024===============
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=25, batch_size=1024, verbose=0)
#===============Compile and train the model for 25 epochs, with a batch size of 1024===============


#===============Save the model===============
# Saving model and weights as HDF5.
model.save('model_weights.hdf5')
# Loading model and weights as HDF5.
loaded_model = load_model('model_weights.hdf5')
# Predicting using loaded model.
evaluate(loaded_model, X_test, y_test)
#===============Save the model===============