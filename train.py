# Author: Adrian Golias
# Importing Libraries
import numpy as np
# Setting up environment to keep using Keras TensorFlow backend
import os
# For CPU testing
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# For GPU testing
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Importing Keras datasets & essential libraries to build neural network
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils

# Display message to inform user that system will load dataset
print("Loading MNIST Dataset...")
# Loading MNIST dataset onto the system, which will automatically download the dataset from
# the following location: https://s3.amazonaws.com/img-datasets/mnist.npz if not already downloaded
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# Display message to inform user that loading is done
print("Loading Complete.")

# Displaying total number of shapes before reshaping and normalizing
print("\nDisplaying total # of shapes in each category:")
print("Total # for X-Train Shapes", X_train.shape)
print("Total # for Y-Train Shapes", y_train.shape)
print("Total # for X-Test Shapes", X_test.shape)
print("Total # for Y-Test Shapes", y_test.shape)

# Reshaping the 28x28 pixel inputs into a single vector before normalization
print("\nReshaping inputs...")
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")

# Normalization of pixel values to lie between 0 & 1. Normalization of  
# data helps to speed up the training & reduces the chance of getting stuck
print("Normalizing Data...")
X_train /= 255
X_test /= 255

# Displaying final shapes ready for training
print("\nDisplaying final shapes ready for training:")
print("Train matrix shape", X_train.shape)
print("Test matrix shape", X_test.shape)

# Checking if Y for machine learning still holds integer values from 0 to 9
print("\nChecking integers values & it's amount...")
print(np.unique(y_train, return_counts=True))

# Defining amount of classes
n_classes = 10
# One-Hot encoding using keras' numpy-related utilities, which converts 
# variables into a form which can be provided to Machine Learning algorithms
# to make more accurate predictions. Adapted from: 
# https://hackernoon.com/what-is-one-hot-encoding-why-and-when-do-you-have-to-use-it-e3c6186d008f
print("\n# of shapes before one-hot encoding: ", y_train.shape)
Y_train = np_utils.to_categorical(y_train, n_classes)
Y_test = np_utils.to_categorical(y_test, n_classes)
print("# of shapes after one-hot encoding: ", Y_train.shape)

# Creating a linear stack of layers with sequential model package from keras,
# which allows us to create a neural network
# Adapted from: https://keras.io/getting-started/sequential-model-guide/
print("\nCreating neural network...")
model = Sequential()
model.add(Dense(512, input_shape=(784,)))
model.add(Activation('relu'))                            
model.add(Dropout(0.2))

model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(10))
model.add(Activation('softmax'))

# Compiling keras sequential model with accuracy metrics and adam optimizer 
# since values have been normalised to lie between 0 to 1
# Optimizer adapted from: https://keras.io/optimizers/
print("\nCompiling sequential model...")
model.compile(loss="categorical_crossentropy", metrics=["accuracy"], optimizer="adam")

# Training the model & saving metrics in history
# https://keras.io/models/model/
print("Initiating model training...")
history = model.fit(X_train, Y_train,
          batch_size=128, epochs=8,
          verbose=2,
          validation_data=(X_test, Y_test))
print("Model training complete.")

# Saving the model to custom directory
# https://keras.io/getting-started/faq/
print("Saving results...")
# NOTE! -> save_dir should be changed acordingly to to match 
# individuals OS system files
save_dir = "c:/Users/Adrian/Documents/GitHub/Handwritten-Digit-Recognition/"
model_name = "model.h5"
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print("Trained model saved to %s " % model_path)