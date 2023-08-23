"""
CONTACT:
Hasan Al-Oqool : https://www.linkedin.com/in/hasanoqool/
"""

#===============Import the required libraries===============
import glob
import os
import tarfile
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
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


#===============Load all image paths and print # of images found===============
data_pattern = os.path.sep.join([data_directory, '*/*/*.png'])
image_paths = list(glob.glob(data_pattern))
print(f'There are {len(image_paths):,} images in the dataset')
#===============Load all image paths and print # of images found===============


#===============Load a single image from the dataset===============
sample = load_img(image_paths[6])
print(f'type: {type(sample)}')
print(f'format: {sample.format}')
print(f'mode: {sample.mode}')
print(f'size: {sample.size}')
#===============Load a single image from the dataset===============


#===============Convert an image into a NumPy array and plot it===============
image_array = img_to_array(sample)
print(f'type: {type(image_array)}')
print(f'shape: {image_array.shape}')
#Display an image using matplotlib after normalization
plt.imshow(image_array / 255.0)
#===============Convert an image into a NumPy array and plot it===============


#===============Load a batch of images using ImageDataGenerator===============
image_generator = ImageDataGenerator(rescale=1.0 / 255.0)
iterator = (image_generator.flow_from_directory(directory=data_directory,batch_size=15))
for batch, _ in iterator:
    plt.figure(figsize=(5, 5))
    for index, image in enumerate(batch, start=1):
        ax = plt.subplot(5, 5, index)
        plt.imshow(image)
        plt.axis('off')
    plt.show()
    break
#===============Load a batch of images using ImageDataGenerator===============