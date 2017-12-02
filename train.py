# Author: Adrian Golias
# Importing Libraries
import numpy as np
# Setting up environment to keep using Keras TensorFlow backend
import os
# For CPU testing
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# For GPU testing
#os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Importing Keras datasets & essential libraries to build neural network
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils

# Display message to inform user that system will load dataset
print("Loading MNIST Dataset...")
# Loading MNIST dataset onto the system, which will automatically download the dataset from
# the following location: https://s3.amazonaws.com/img-datasets/mnist.npz if not already downloaded
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
# Display message to inform user that loading is done
print("Loading Complete.")

# Displaying total number of shapes before reshaping and normalizing
print("\nDisplaying total # of shapes in each category:")
print("Total # for X-Train Shapes", X_train.shape)
print("Total # for Y-Train Shapes", Y_train.shape)
print("Total # for X-Test Shapes", X_test.shape)
print("Total # for Y-Test Shapes", Y_test.shape)

# Reshaping the 28x28 pixel inputs into a single vector before normalizing it
print("\nReshaping inputs...")
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')