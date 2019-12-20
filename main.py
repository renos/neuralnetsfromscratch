import sys
from utils import *
from layers import *
from network import *
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
  print(Xavier_Initialization((5,10),10).shape)
  
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

  x_train = x_train.reshape((60000, 1, 28, 28))
  x_test = x_test.reshape((10000, 1, 28, 28))
  '''
  nnetwork = NeuralNetwork(
      Convolution(F=6, C=1, HH=5, WW=5, padding=2), ReLU(),
      MaxPool(HH=2, WW=2, stride=2),
      Convolution(F=16, C=6, HH=5, WW=5), ReLU(),
      MaxPool(HH=2, WW=2, stride=2),
      Convolution(F=120, C=16, HH=5, WW=5), ReLU(),
      Flatten(),
      Linear(in_dimension=120, out_dimension=84), ReLU(),
      Linear(in_dimension=84, out_dimension=10),
      SoftmaxLoss()
  )
'''

  nnetwork = NeuralNetwork(
      Flatten(),
      Linear(in_dimension=784, out_dimension=196), ReLU(),
      Linear(in_dimension=196, out_dimension=84), ReLU(),
      Linear(in_dimension=84, out_dimension=10),
      SoftmaxLoss())
  train_count = 60000
  test_count = 10000

  if sys.argv[1] == "adam":
    name = "adam"
    optim = Adam()
  elif sys.argv[1] == "momentum":
    name = "momentum"
    optim = Momentum(learn_rate=0.001, gamma=0.9)
  elif sys.argv[1] == "adagrad":
    name = "adagrad"
    optim = AdaGrad()
  elif sys.argv[1] == "sgd":
    name = "sgd"
    optim = GradientDescent(learn_rate=0.001)
  else:
    print("invalid optimizer")
  print(name, "optimizer")

  results = []

  for i in range(100):
    print("Epoch", i)
    nnetwork.train(x_train[:train_count], y_train[:train_count],
        optim, 1, batch_size=32)
    y_predict = nnetwork.predict(x_test[:test_count])
    print(y_predict)
    acc = np.mean(y_predict == y_test[:test_count])
    print("Test set accuracy:", acc)
    results.append(acc)

  print(name, "by epoch:")
  print(results)

