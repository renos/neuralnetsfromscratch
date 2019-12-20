import numpy as np
from layers import *
from utils import *


class NeuralNetwork():
  
  def __init__(self, *layers):
    self.layers = list(*layers)

  def test(self, x):
    if x.ndim == 2:
      out = x
    else:
      out = x[np.newaxis,:]

    for layer in self.layers:
      out = layer.forward(out)
    return out

  def train(self, x_train, optim, num_epochs, batch_size=None):
    num_samples, _ = x_train.shape
    if batch_size is None:
      batch_size = num_samples
    for epoch in range(num_epochs):
      start = 0
      while start < num_samples:
        end = min(num_samples, start + batch_size)
        x = x_train[start:end]
        out = x
        # forward pass
        for layer in self.layers:
          out = layer.forward(out)
        # backward pass
        dout = out
        for layer in reversed(self.layers):
          dout, *dweights = layer.backward(dout)
          optim.update(layer, *dweights)
        start += batch_size
      optim.end_epoch()

class GradientDescent():

  def __init__(self, learn_rate=0.001):
    self.learn_rate = learn_rate
    self.t = 0

  def update(self, layer, *dweights):
    if dweights:
      diff = list(map(lambda w: -1 * self.learn_rate * w, dweights))
      layer.update_weights(*diff)

  def end_epoch(self):
    self.t += 1


class Momentum():

  def __init__(self, learn_rate=0.001, gamma=0.9):
    self.learn_rate = learn_rate
    self.gamma = gamma
    self.cache = {}
    self.t = 0

  def update(self, layer, *dweights):
    if dweights:
      if layer not in self.cache:
        self.cache[layer] = [0] * len(dweights)
      self.cache[layer] = list(map(
          lambda ws: self.learn_rate * ws[0] + self.gamma * ws[1],
          zip(dweights, self.cache[layer])))
      diff = list(map(lambda w: -1 * w, self.cache[layer]))
      layer.update_weights(*diff)

  def end_epoch(self):
    self.t += 1

class AdaGrad():

  def __init__(self, learn_rate=0.001):
    self.learn_rate = learn_rate
    self.cache = {}
    self.t = 0

  def update(self, layer, *dweights):
    if dweights:
      if layer not in self.cache:
        self.cache[layer] = [0] * len(dweights)
      self.cache[layer] = list(map(
          lambda ws: pow(np.linalg.norm(ws[0]), 2) + ws[1],
          zip(dweights, self.cache[layer])))
      diff = list(map(
          lambda w: -1 * self.learn_rate / np.sqrt(w[1] + 1e-8) * w[0],
          zip(dweights, self.cache[layer])))
      layer.update_weights(*diff)

  def end_epoch(self):
    self.t += 1

class Adam():

  def __init__(self, learn_rate=0.001, beta1=0.9, beta2=0.999):
    self.learn_rate = learn_rate
    self.beta1 = beta1
    self.beta2 = beta2
    self.cache = {}
    self.t = 0

  def update(self, layer, *dweights):
    if dweights:
      if layer not in self.cache:
        self.cache[layer]["m"] = [0] * len(dweights)
        self.cache[layer]["v"] = [0] * len(dweights)
      self.cache[layer]["m"] = list(map(
          lambda ws: self.beta1 * ws[1] + (1 - self.beta1) * ws[0],
          zip(dweights, self.cache[layer]["m"])))
      self.cache[layer]["v"] = list(map(
          lambda ws: self.beta2 * ws[1] + (1 - self.beta2) * np.square(ws[0]),
          zip(dweights, self.cache[layer]["m"])))
      m_hat = list(map(
          lambda m: m / (1 - pow(self.beta1, self.t)), self.cache[layer]["m"]))
      v_hat = list(map(
          lambda v: v / (1 - pow(self.beta2, self.t)), self.cache[layer]["v"]))
      diff = list(map(
          lambda ws: -1 * self.learn_rate * ws[0] / (np.sqrt(ws[1]) + 1e-8),
          zip(m_hat, v_hat)))
      layer.update_weights(*diff)

  def end_epoch(self):
    self.t += 1


# def train(args, model, device, train_loader, optimizer, epoch):
#   model.train()
#   for batch_idx, (data, target) in enumerate(train_loader):
#     data, target = data.to(device), target.to(device)
#     optimizer.zero_grad()
#     output = model(data)
#     print(output)
#     print(target)
#     loss = F.nll_loss(output, target)
#     loss.backward()
#     optimizer.step()
#     if batch_idx % args.log_interval == 0:
#       print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#         epoch, batch_idx * len(data), len(train_loader.dataset),
#         100. * batch_idx / len(train_loader), loss.item()))
