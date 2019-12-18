import numpy as np
from utils import *


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
  def __init__(self, in_dimension, out_dimension, bias=False, activation='ReLU'):
    self.in_dimension = in_dimension
    self.out_dimension = out_dimension
    self.use_bias = bias
    self.activation = activation
    self.init_params()
  
  def init_params(self):
    in_dimension = self.in_dimension
    out_dimension = self.out_dimension
    if(self.activation == 'ReLU'):
      self.weight = He_Initialization(
          (in_dimension, out_dimension), out_dimension)
      if(self.use_bias):
        self.bias = np.zeros(out_dimension)
    else:
      self.weight = Xavier_Initialization(
          (in_dimension, out_dimension), out_dimension)
      if(self.use_bias):
        self.bias = np.zeros(out_dimension)

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
    return dx, dweight, dbias

  def updateweight(self, weight):
    self.weight = weight

  def updatebias(self,bias): 
    self.bias = bias


class MaxPooling():
	def __init__(self, C,HH,WW):



class Convolutional():

  def __init__(self, F, C, HH, WW, bias=False, padding=0, stride=1, activation='ReLU'):
    self.in_channels = C
    self.out_channels = F
    self.filter_height = HH
    self.filter_width = WW
    self.padding = padding
    self.stride = stride

  def init_params(self):
    width = self.filter_width
    height = self.filter_height
    in_ch = self.in_channels
    out_ch = self.out_channels
    if(self.activation == 'ReLU'):
      self.weight = He_Initialization(
          (out_ch, in_ch, height, width), in_ch * width * height)
      if(self.use_bias):
        self.bias = np.zeros(self.out_ch)
    else:
      self.weight = Xavier_Initialization(
          (out_ch, in_ch, height, width), in_ch * width * height)
      if(self.use_bias):
        self.bias = np.zeros(self.out_ch)

  def forward(self, x):
    # input: x of shape (N, C, H, W)
    # filter self.wight is of shape (F, C, HH, WW)
    # returns: out of shape (N, F, H', W')
    batch_size, in_ch, height, width = x.shape
    out_ch, _, filter_height, filter_width = self.weight.shape
    padding = self.padding
    stride = self.stride
    weight = self.weight
    bias = self.bias

    padded = np.pad(x, ((0,0), (0,0), (padding,padding), (padding,padding)),
        'constant', constant_values=0)
    H_prime = int(1 + (height + 2 * padding - filter_height) / stride)
    W_prime = int(1 + (width + 2 * padding - filter_width) / stride)
    out = np.zeros((batch_size, out_ch, H_prime, W_prime))

    for sample_index in range(batch_size):
      for filter_index in range(out_ch):
        for h_pos in range(H_prime):
          for w_pos in range(W_prime):
             h_start, h_end = (h_pos * stride, h_pos * stride + filter_height)
             w_start, w_end = (w_pos * side, w_pos * stride + filter_width)
             conv_slice = padded[sample_index, :, \
                 h_start : h_end, w_start : w_end]
             conv_sum = np.sum(conv_slice * weight[filter_index])
             out[sample_index, filter_index, h_pos, w_pos] = conv_sum \
                 + bias[filter_index]

    self.cache = (x, weight, bias)
    return out

   def backward(self, dout):

