import numpy as np




class ReLU():
  def __init__(self):
    cache = None

  def forward(self, x):
    cache = x
    return np.maximum(x,0)

  def backward(self, dout):
    x = cache
    return dout * np.piecewise(x, [x <= 0, x > 0], [0, 1])

class Linear():
  def __init__(self, in_dimension, out_dimension, bias = False, activation = 'ReLU'):
    self.in_dimension = in_dimension
    self.out_dimension = out_dimension
    self.use_bias = bias
    self.activation = activation
    self.init_params()
  
  def init_params(self):
    if(self.activation == 'ReLU'):
      self.weight = He_Initialization(self.in_dimension,self.out_dimension)
      if(self.use_bias):
        self.bias = np.zeros(self.out_dimension)
    else:
      self.weight = Xavier_Initialization(self.in_dimension,self.out_dimension)
      if(self.use_bias):
        self.bias = np.zeros(self.out_dimension)

  def forward(self, x):
    xW = x.dot(self.weight)
    out = xW + self.bias
    self.cache = x
    return out

  def backward(self, dout):
    x = self.cache
    dx = dout.dot((self.weight.T))
    dweight = (x.T).dot(dout)
    dbias = np.add.reduce(dout, axis = 0)
    return dx,dweight,dbias

  def updateweight(self, weight):
    self.weight = weight

  def updatebias(self,bias): 
    self.bias = bias


class Convolutional():

  def __init__(self, F, C, HH, WW, bias = False, activation = 'ReLU'):
    self.in_channels = C
    self.out_channels = F
    self.filter_height = HH
    self.filter_width = WW


  def init_params(self):
    if(self.activation == 'ReLU'):
      self.weight = He_Initialization(self.in_dimension,self.out_dimension)
      if(self.use_bias):
        self.bias = np.zeros(self.out_dimension)
    else:
      self.weight = Xavier_Initialization(self.in_dimension,self.out_dimension)
      if(self.use_bias):
        self.bias = np.zeros(self.out_dimension)

  




