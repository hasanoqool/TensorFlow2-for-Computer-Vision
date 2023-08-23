"""
CONTACT:
Hasan Al-Oqool : https://www.linkedin.com/in/hasanoqool/
"""

#===============Import the required libraries===============
import os
import tarfile
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.utils import get_file
#===============Import the required libraries===============


#===============Define the URL of the CINIC-10 dataset===============
Dataset_URL = 'https://datashare.is.ed.ac.uk/bitstream/handle/10283/3192/CINIC-10.tar.gz?sequence=4&isAllowed=y'
Data_Name = 'cinic10'
File_Extension = 'tar.gz'
File_Name = '.'.join([Data_Name, File_Extension])
#===============Define the URL of the CINIC-10 dataset===============


#===============Download and decompress the data===============
downloaded_file_location = get_file(origin=Dataset_URL, fname=File_Name, extract=False)
# Build the path to the data directory based on the location of the downloaded file.
data_directory, _ = downloaded_file_location.rsplit(os.path.sep, maxsplit=1)
data_directory = os.path.sep.join([data_directory, Data_Name])
# Only extract the data if it hasn't been extracted already
if not os.path.exists(data_directory):
    tar = tarfile.open(downloaded_file_location)
    tar.extractall(data_directory)
#===============Download and decompress the data===============


#===============Create a dataset of image paths using tf.data.Dataset===============
data_pattern = os.path.sep.join([data_directory, '*/*/*.png'])
image_dataset = tf.data.Dataset.list_files(data_pattern)
#===============Create a dataset of image paths using tf.data.Dataset===============


#===============Take a single path from the dataset and plot it===============
for file_path in image_dataset.take(1):
    sample_path = file_path.numpy()
sample_image = tf.io.read_file(sample_path)
#we must convert it into a format a neural network can work with
sample_image = tf.image.decode_png(sample_image, channels=3)
sample_image = sample_image.numpy()
#Display an image using matplotlib after normalization
plt.imshow(sample_image / 255.0)
#===============Take a single path from the dataset and plot it===============


'''Take the first 10 elements of image_dataset, 
decode and normalize them, and
then display them using matplotlib'''
plt.figure(figsize=(5, 5))
for index, image_path in enumerate(image_dataset.take(10), start=1):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.convert_image_dtype(image, np.float32)
    ax = plt.subplot(5, 5, index)
    plt.imshow(image)
    plt.axis('off')
plt.show()
plt.close()
'''Take the first 10 elements of image_dataset, 
decode and normalize them, and
then display them using matplotlib'''

