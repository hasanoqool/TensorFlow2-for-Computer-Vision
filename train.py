"""
CONTACT:
Hasan Al-Oqool : https://www.linkedin.com/in/hasanoqool/
"""

#===============Import the required libraries===============
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras import Input
from tensorflow.keras.datasets import mnist #0-9 digits (10 classes) 28*28*1CH
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model, Sequential
#===============Import the required libraries===============


#===============Create a model using the Sequential API===============
layers = [
    Dense(256, input_shape=(28 * 28 * 1, ), activation="relu"),
    Dense(128, activation="relu"),
    Dense(10, activation="softmax")
]
Sequential_list = Sequential(layers)
#===============Create a model using the Sequential API===============


#===============Create a model using add()===============
Sequential_add = Sequential()
Sequential_add.add(Dense(256, input_shape=(28 * 28 * 1, ), activation="relu"))
Sequential_add.add(Dense(128, activation="relu"))
Sequential_add.add(Dense(10, activation="softmax"))
#===============Create a model using add()===============


#===============Create a model using Functional API===============
input_layer = Input(shape=(28 * 28 * 1,))
dense1 = Dense(256, activation='relu')(input_layer)
dense2 = Dense(128, activation='relu')(dense1)
predictions = Dense(10, activation='softmax')(dense2)
functional_model = Model(inputs=input_layer, outputs=predictions)
#===============Create a model using Functional API===============


#===============Create a model using an OOP approach by sub-classing tensorflow===============
class ClassModel (Model):
    def __init__(self):
        super(ClassModel, self).__init__()
        self.dense1 = Dense(256, activation='relu')
        self.dense2 = Dense(128, activation='relu')
        self.predicitons = Dense(10, activation="softmax")

    def call(self, inputs, **kwargs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.predicitons(x)
    
class_model = ClassModel()
#===============Create a model using an OOP approach by sub-classing tensorflow===============


#===============Prepare the MNIST data into vector format===============
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape((X_train.shape[0], 28 * 28 * 1)) #from (60000,28,28) to (60000,784)
X_test = X_test.reshape((X_test.shape[0], 28 * 28 * 1)) #from (10000,28,28) to (10000,784)
X_train = X_train.astype('float32') / 255.0 #normalization
X_test = X_test.astype('float32') / 255.0 #normalization
#===============Prepare the MNIST data into vector format===============


#===============One-hot encode the labels===============
label_binarizer = LabelBinarizer()
y_train = label_binarizer.fit_transform(y_train)
y_test = label_binarizer.fit_transform(y_test)
#===============One-hot encode the labels===============


#===============Take 20% of the data for validation===============
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2)
#===============Take 20% of the data for validation===============


#===============Compile, train the models for 25 epochs, and evaluate them on the test set===============
models = {
'sequential_model': Sequential_add,
'sequential_list': Sequential_list,
'functional_model': functional_model,
'class_model': class_model
}

for name, model in models.items():

    print(f'Compiling model: {name}')
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    print(f'Training model: {name}')
    model.fit(X_train, y_train,validation_data=(X_valid, y_valid), epochs=25, batch_size=256, verbose=0)
    _, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f'Testing model: {name}. \nAccuracy: {accuracy}')    
    print('---')
#===============Compile, train the models for 25 epochs, and evaluate them on the test set===============