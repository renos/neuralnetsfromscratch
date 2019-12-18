from utils import *
from layers import *
import numpy as np

try:
  with open("mnist.pkl",'rb') as f:
    mnist = pickle.load(f)
    x_train = mnist["training_images"]
    y_train = mnist["training_labels"]
    x_test = mnist["test_images"]
    y_test = mnist["test_labels"]
except IOError:
    download_mnist()
    with open("mnist.pkl",'rb') as f:
      mnist = pickle.load(f)
      x_train = mnist["training_images"]
      y_train = mnist["training_labels"]
      x_test = mnist["test_images"]
      y_test = mnist["test_labels"]



