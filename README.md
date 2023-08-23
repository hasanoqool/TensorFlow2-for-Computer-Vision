# TensorFlow2-for-Computer-Vision
Working with Keras API and Tensorflow on different datasets

## Running APIs.py
* Create four models by working with different blocks of the Keras API on <b>MNIST dataset</b>.

| Model  |  Accuracy |
| ------------- | ------------- |
| sequential_model  | 0.9810000061988831  |
| sequential_list  | 0.9807999730110168  |
| functional_model  | 0.979200005531311  |
| class_model  | 0.9797000288963318  |


## Running load_keras.py
* Display an image using <b>matplotlib</b> after normalization.

![Sample](https://github.com/hasanoqool/TensorFlow2-for-Computer-Vision/blob/main/images/boat.png)



* Load a batch of images using <b>ImageDataGenerator</b>.

![Batch](https://github.com/hasanoqool/TensorFlow2-for-Computer-Vision/blob/main/images/multi.png)


## Running load_tf.py
* Take a single path from the dataset and plot it.

![Sample2](https://github.com/hasanoqool/TensorFlow2-for-Computer-Vision/blob/main/images/frog.png)



* Take the first 10 elements of image_dataset, decode and normalize them, and then display them using <b>matplotlib</b>.

![Batch2](https://github.com/hasanoqool/TensorFlow2-for-Computer-Vision/blob/main/images/multi2.png)


## Running saving_loading_model.py
* Saving both the model and its weights in a single <b>HDF5</b> file using the <b>save() method</b>.
* Trained and Tested on <b>MNIST dataset</b>.

| Model  | Accuracy |
| ------------- | ------------- |
| model_weights  | 0.98  |


