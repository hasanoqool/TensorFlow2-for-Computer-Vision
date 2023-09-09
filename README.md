# TensorFlow2-for-Computer-Vision
Working with Keras API and Tensorflow on different datasets
#

# Overview
* This repo consists of 6 main topics:
    * Building model blocks of the Keras API
    * Loading images using the Keras API
    * Loading images using the tf.data.Dataset API
    * Saving and loading a model
    * Visualizing a model's architecture
    * Creating image classifier
#

## Running APIs.py
* Create four models by working with different blocks of the Keras API on <b>MNIST dataset</b>.

| Model  |  Accuracy |
| ------------- | ------------- |
| sequential_model  | 0.9810000061988831  |
| sequential_list  | 0.9807999730110168  |
| functional_model  | 0.979200005531311  |
| class_model  | 0.9797000288963318  |

#
## Running load_keras.py
* Display an image using <b>matplotlib</b> after normalization.

![Sample](https://github.com/hasanoqool/TensorFlow2-for-Computer-Vision/blob/main/images/boat.png)



* Load a batch of images using <b>ImageDataGenerator</b>.

![Batch](https://github.com/hasanoqool/TensorFlow2-for-Computer-Vision/blob/main/images/multi.png)

#
## Running load_tf.py
* Take a single path from the dataset and plot it.

![Sample2](https://github.com/hasanoqool/TensorFlow2-for-Computer-Vision/blob/main/images/frog.png)



* Take the first 10 elements of image_dataset, decode and normalize them, and then display them using <b>matplotlib</b>.

![Batch2](https://github.com/hasanoqool/TensorFlow2-for-Computer-Vision/blob/main/images/multi2.png)

#
## Running saving_loading_model.py
* Saving both the model and its weights in a single <b>HDF5</b> file using the <b>save() method</b>.
* Trained and Tested on <b>MNIST dataset</b>.

| Model  | Accuracy |
| ------------- | ------------- |
| model_weights  | 0.98  |

#
## Running model_architecture.py
* Two different ways we can display a <b>model's architecture</b>:

    • Using a <b>text summary</b>.

    ![summary](https://github.com/hasanoqool/TensorFlow2-for-Computer-Vision/blob/main/images/model_summary.png)


    • Using a <b>visual diagram</b>.

    ![plot](https://github.com/hasanoqool/TensorFlow2-for-Computer-Vision/blob/main/images/model_arch.jpg)

#
## Running image_classifier.py
* implementing an image classifier on <b>Fashion-MNIST</b>:

    • <b>loss curve</b> both on the training and validation sets.

    ![1](https://github.com/hasanoqool/TensorFlow2-for-Computer-Vision/blob/main/images/image_classifier/loss.png)


    • <b>accuracy curve</b> for the training and validation sets.

    ![2](https://github.com/hasanoqool/TensorFlow2-for-Computer-Vision/blob/main/images/image_classifier/accuracy.png)


    • <b>model architecture</b> of our model.

    ![3](https://github.com/hasanoqool/TensorFlow2-for-Computer-Vision/blob/main/images/image_classifier/model.png)
#
## Contact
* Reach me out here: https://www.linkedin.com/in/hasanoqool/
