import sys
from utils import *
from layers import *
from network import *
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
  
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

  nnetwork = NeuralNetwork(
      Flatten(),
      Linear(in_dimension=784, out_dimension=196), ReLU(),
      Linear(in_dimension=196, out_dimension=84), ReLU(),
      Linear(in_dimension=84, out_dimension=10),
      SoftmaxLoss()
  )

  train_count = 60000
  test_count = 10000

  learn_rate = float(sys.argv[2])
  batch_size = int(sys.argv[3])
  lam = float(sys.argv[4])

  if sys.argv[1] == "adam":
    name = "adam"
    optim = Adam(learn_rate=learn_rate, lam=lam)
  elif sys.argv[1] == "momentum":
    name = "momentum"
    gamma = float(sys.argv[5])
    print("gamma", gamma)
    optim = Momentum(learn_rate=learn_rate, lam=lam, gamma=gamma)
  elif sys.argv[1] == "adagrad":
    name = "adagrad"
    optim = AdaGrad(learn_rate=learn_rate, lam=lam)
  elif sys.argv[1] == "sgd":
    name = "sgd"
    optim = GradientDescent(learn_rate=learn_rate, lam=lam)
  else:
    print("invalid optimizer")
  print(name, "learn_rate", learn_rate, "batch_size", batch_size,
      "lam", lam)

  results = []
  losses = []

  for i in range(100):
    print("Epoch", i)
    batch_losses = nnetwork.train(x_train[:train_count], y_train[:train_count],
        optim, 1, batch_size=batch_size)
    losses.append(np.mean(batch_losses))
    y_predict = nnetwork.predict(x_test[:test_count])
    acc = np.mean(y_predict == y_test[:test_count])
    print("Test set accuracy:", acc)
    results.append(acc)
    print("Loss:", losses[-1])

  print(name, "Test set accuracy:")
  print(results)
  print("Losses:")
  print(losses)

