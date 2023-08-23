"""
CONTACT:
Hasan Al-Oqool : https://www.linkedin.com/in/hasanoqool/
"""

#===============Import the required libraries===============
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_docs as tfdocs
import tensorflow_docs.plots
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from keras.models import Model, load_model
from keras.utils import plot_model
from keras.layers import *
from keras.datasets import fashion_mnist as fm
from keras.models import Sequential
#===============Import the required libraries===============


#===============Define a function that will load and prepare the dataset===============
def load_dataset():
    (X_train, y_train), (X_test, y_test) = fm.load_data()
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0

    # Reshape grayscale to include channel dimension.
    X_train = np.expand_dims(X_train, axis=3)
    X_test = np.expand_dims(X_test, axis=3)

    label_binarizer = LabelBinarizer()
    y_train = label_binarizer.fit_transform(y_train)
    y_test = label_binarizer.fit_transform(y_test)

    (X_train, X_val, y_train, y_val) = train_test_split(X_train, y_train,test_size=0.2)
    train_ds = (tf.data.Dataset.from_tensor_slices((X_train,y_train)))
    val_ds = (tf.data.Dataset.from_tensor_slices((X_val, y_val)))
    test_ds = (tf.data.Dataset.from_tensor_slices((X_test, y_test)))
    return train_ds, val_ds, test_ds
#===============Define a function that will load and prepare the dataset===============


#===============function for building network===============
def build_network():
    model = Sequential()
    #block1  
    model.add(Input(shape=(28, 28, 1), name='input_layer'))
    model.add(Conv2D(kernel_size=(5, 5), padding='same', strides=(1, 1), filters=20, activation="elu", name='convolution_1'))
    model.add(BatchNormalization(name='batchnormalization_1'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='pooling_1'))
    model.add(Dropout(0.5, name='drop_out1'))
    #block2
    model.add(Conv2D(kernel_size=(5, 5), padding='same', strides=(1, 1), filters=20, activation="elu", name='convolution_2'))
    model.add(BatchNormalization(name='batchnormalization_2'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='pooling_2'))
    model.add(Dropout(0.5, name='drop_out2'))
    #block3
    model.add(Flatten(name='flatten'))
    model.add(Dense(units=500, activation="elu", name='dense_1'))
    model.add(Dense(units=10, activation="softmax", name='output'))
    return model
#===============function for building network===============


#===============Define a function that takes a model's training history===============
def plot_model_history(model_history, metric, ylim=None):
    plt.style.use('seaborn-darkgrid')
    plotter = tfdocs.plots.HistoryPlotter()
    plotter.plot({'Model': model_history}, metric=metric)
    plt.title(f'{metric.upper()}')
    if ylim is None:
        plt.ylim([0, 1])
    else:
        plt.ylim(ylim)
        plt.savefig(f'{metric}.png')
        plt.close()
#===============Define a function that takes a model's training history===============


#===============Consume the training and validation datasets in batches of 256 images at a time===============
BATCH_SIZE = 256
BUFFER_SIZE = 1024
train_dataset, val_dataset, test_dataset = load_dataset()
train_dataset = (train_dataset.shuffle(buffer_size=BUFFER_SIZE).batch(BATCH_SIZE).prefetch(buffer_size=BUFFER_SIZE))
val_dataset = (val_dataset.batch(BATCH_SIZE).prefetch(buffer_size=BUFFER_SIZE))
test_dataset = test_dataset.batch(BATCH_SIZE)
#===============Consume the training and validation datasets in batches of 256 images at a time===============


#===============Build and train the network===============
EPOCHS = 25
model = build_network()
model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
model_history = model.fit(train_dataset, validation_data=val_dataset, epochs=EPOCHS, verbose=1)
#===============Build and train the network===============


#===============Plot the training and validation loss and accuracy===============
plot_model_history(model_history, 'loss', [0., 2.0])
plot_model_history(model_history, 'accuracy')
#===============Plot the training and validation loss and accuracy===============


#===============Visualize the model's architecture===============
plot_model(model, show_shapes=True, show_layer_names=True, to_file='model.png')
#===============Visualize the model's architecture===============


#===============Save and evaluate model===============
model.save('image_classifier.hdf5')

loaded_model = load_model('image_classifier.hdf5')
results = loaded_model.evaluate(test_dataset, verbose=0)
print(f'Loss: {results[0]}, Accuracy: {results[1]}')
#===============Save and evaluate model===============

