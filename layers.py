import numpy as np
from utils import *


class ReLU():

  def __init__(self):
    self.cache = None

  def forward(self, x, y=None):
    self.cache = x
    return np.maximum(x,0)

  def backward(self, dout):
    x = self.cache
    return (dout * np.piecewise(x, [x <= 0, x > 0], [0, 1]),)


class SoftmaxLoss():

  def __init__(self):
    self.cache = None

  def forward(self, x, y):
    x_shift = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(x_shift), axis=1, keepdims=True)
    nll = x_shift - np.log(Z)
    probs = np.exp(nll)
    N = x.shape[0]
    loss = -np.sum(nll[np.arange(N), y]) / N
    self.cache = probs, y, N
    return loss

  def backward(self, dout):
    probs, y, N = self.cache
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return (dx,)


class Linear():
  
  def __init__(self, in_dimension, out_dimension, bias=False, activation='ReLU'):
    self.in_dimension = in_dimension
    self.out_dimension = out_dimension
    self.use_bias = bias
    self.bias = np.zeros(out_dimension)
    self.activation = activation
    self.init_params()
    self.cache = None
  
  def init_params(self):
    in_dimension = self.in_dimension
    out_dimension = self.out_dimension
    if self.activation == 'ReLU':
      self.weight = He_Initialization(
          (in_dimension, out_dimension), out_dimension)
    else:
      self.weight = Xavier_Initialization(
          (in_dimension, out_dimension), out_dimension)

  def forward(self, x, y=None):
    xW = x.dot(self.weight)
    out = xW + self.bias
    self.cache = x
    return out

  def backward(self, dout):
    x = self.cache
    dx = dout.dot((self.weight.T))
    dweight = (x.T).dot(dout)
    if self.use_bias:
      dbias = np.add.reduce(dout, axis = 0)
    else:
      dbias = self.bias #zeros
    return (dx, dweight, dbias)

  def update_weights(self, dweight, dbias):
    self.weight += dweight
    self.bias += dbias


class MaxPool():

  def __init__(self, HH, WW, stride=2):
    self.filter_height = HH
    self.filter_width = WW
    self.stride = stride
    self.cache = None

  def forward(self, x, y=None):
    # input: x of shape (N, F, H, W)
    # returns: out of shape (N, F, H', W')
    batch_size, in_ch, height, width = x.shape
    filter_height = self.filter_height
    filter_width = self.filter_width
    stride = self.stride
    H_prime = int(1 + (height - filter_height) / stride)
    W_prime = int(1 + (width - filter_width) / stride)
    
    out = np.zeros((batch_size, in_ch, H_prime, W_prime))
 
    for sample_index in range(batch_size):
      for channel_index in range(in_ch):
        for h_pos in range(H_prime):
          for w_pos in range(W_prime):
             h_start, h_end = (h_pos * stride, h_pos * stride + filter_height)
             w_start, w_end = (w_pos * stride, w_pos * stride + filter_width)
             maxpool_slice = x[sample_index, :, \
                 h_start : h_end, w_start : w_end]
             out[sample_index, channel_index, h_pos, w_pos] \
                 = np.amax(maxpool_slice)

    self.cache = x
    return out

  def backward(self, dout):
    # input: dout of shape (N, F, H', W')
    # returns: dx of shape (N, F, H, W)\
    batch_size, in_ch, H_prime, W_prime = dout.shape
    x = self.cache
    _, _, height, width = x.shape
    filter_height = self.filter_height
    filter_width = self.filter_width
    stride = self.stride

    dx = np.zeros(x.shape)
   
    for sample_index in range(batch_size):
      for channel_index in range(in_ch):
        for h_pos in range(H_prime):
          for w_pos in range(W_prime):
             h_start, h_end = (h_pos * stride, h_pos * stride + filter_height)
             w_start, w_end = (w_pos * stride, w_pos * stride + filter_width)
             maxpool_slice = x[sample_index, :, \
                 h_start : h_end, w_start : w_end]
             dx_view = dx[sample_index, :, \
                 h_start : h_end, w_start : w_end]
             max_index = np.unravel_index(
                 maxpool_slice.argmax(), maxpool_slice.shape)
             dx_view[max_index] = dout[sample_index, \
                 channel_index, h_pos, w_pos]
    return (dx,)
             

class Convolution():

  def __init__(self, F, C, HH, WW, bias=False, padding=0, stride=1, activation='ReLU'):
    self.in_channels = C
    self.out_channels = F
    self.filter_height = HH
    self.filter_width = WW
    self.padding = padding
    self.stride = stride
    self.use_bias = bias
    self.bias = np.zeros(F)
    self.activation = activation
    self.init_params()
    self.cache = None

  def init_params(self):
    width = self.filter_width
    height = self.filter_height
    in_ch = self.in_channels
    out_ch = self.out_channels
    if self.activation == 'ReLU':
      self.weight = He_Initialization(
          (out_ch, in_ch, height, width), in_ch * width * height)
    else:
      self.weight = Xavier_Initialization(
          (out_ch, in_ch, height, width), in_ch * width * height)

  def forward(self, x, y=None):
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
             w_start, w_end = (w_pos * stride, w_pos * stride + filter_width)
             conv_slice = padded[sample_index, :, \
                 h_start : h_end, w_start : w_end]
             conv_sum = np.sum(conv_slice * weight[filter_index])
             out[sample_index, filter_index, h_pos, w_pos] = conv_sum \
                 + bias[filter_index]

    self.cache = (x, weight, bias)
    return out

  def backward(self, dout):
    # input: dout of shape (N, F, H', W')
    # returns: dx of shape (N, C, H, W)
    #          dw of shape (F, C, WW, HH)
    #          db of shape (F,)
    x, weight, bias = self.cache
    batch_size, in_ch, height, width = x.shape
    out_ch, _, filter_height, filter_width = self.weight.shape
    padding = self.padding
    stride = self.stride
    weight = self.weight
    bias = self.bias
    _, _, H_prime, W_prime = dout.shape
    
    padded = np.pad(x, ((0,0), (0,0), (padding,padding), (padding,padding)),
        'constant', constant_values=0)
    padded_dx = np.zeros(padded.shape)
    dx = np.zeros(x.shape)
    dw = np.zeros(weight.shape)
    db = np.zeros(bias.shape)

    for sample_index in range(batch_size):
      for filter_index in range(out_ch):
        if self.use_bias:
          db[filter_index] += np.sum(dout[sample_index, filter_index])
        for h_pos in range(H_prime):
          for w_pos in range(W_prime):
            h_start, h_end = (h_pos * stride, h_pos * stride + filter_height)
            w_start, w_end = (w_pos * stride, w_pos * stride + filter_width)
            conv_slice = padded[sample_index, :, \
                h_start : h_end, w_start : w_end]
            dw[filter_index] += conv_slice \
                * dout[sample_index, filter_index, h_pos, w_pos]
            padded_dx[sample_index, :, h_start : h_end, w_start : w_end] \
                += weight[filter_index] \
                * dout[sample_index, filter_index, h_pos, w_pos]

    dx = padded_dx[:, :, padding : padding + height, padding : padding + width]
    return (dx, dw, db)

  def update_weights(self, dweight, dbias):
    self.weight += dweight
    self.bias += dbias


class Flatten():

  def __init__(self):
    self.cache = None

  def forward(self, x, y=None):
    # input: x of shape (N, ...)
    # returns: out of shape (N, M)
    N, *shape = x.shape
    self.cache = x
    reshaped = np.reshape(x, (N, np.prod(shape)))
    return reshaped

  def backward(self, dout):
    # input: dout of shape(N, M)
    # returns: dx of shape(N, ...) (the same shape as in forward)
    x = self.cache
    unreshaped = np.reshape(dout, x.shape)
    return (unreshaped,)
    
