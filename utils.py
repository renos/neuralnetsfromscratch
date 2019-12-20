import numpy as np
import gzip
import pickle
from urllib import request

import matplotlib.pyplot as plt

def download_mnist():
    base_url = "http://yann.lecun.com/exdb/mnist/"
    files = [
      ["training_images","train-images-idx3-ubyte.gz"],
      ["test_images","t10k-images-idx3-ubyte.gz"],
      ["training_labels","train-labels-idx1-ubyte.gz"],
      ["test_labels","t10k-labels-idx1-ubyte.gz"]
      ]
    for name in files:
        request.urlretrieve(base_url+name[1], name[1])
    mnist = {}
    for name in files[:2]:
        with gzip.open(name[1], 'rb') as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1,28*28)
    for name in files[-2:]:
        with gzip.open(name[1], 'rb') as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=8)
    with open("mnist.pkl", 'wb') as f:
        pickle.dump(mnist,f)
    print('Download Successful')
  
def print_mnist_image(input_array):
  plt.imshow(input_array.reshape((28,28)), cmap="gray")
  plt.show()

def Xavier_Initialization(shape, divisor):
  # shape is a tuple of dimensions
  #Used for Sigmoid / TanH
  return np.random.normal(0,1,shape) * np.sqrt(1.0/divisor)

def He_Initialization(shape, divisor):
  #https://arxiv.org/pdf/1502.01852v1.pdf
  #Used to initalize ReLU networks.
  return np.random.normal(0,1,shape) * np.sqrt(2.0/divisor)

