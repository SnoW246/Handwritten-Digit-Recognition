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
