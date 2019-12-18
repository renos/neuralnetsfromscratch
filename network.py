from layers import *
import numpy as np



class NerualNetwork():
	def __init__(self, *layers):
		self.layers = list(*layers)

	def test(self, x):
		if(x.ndim == 2):
			out = x
		else:
			out = x[np.newaxis,:]

		for layer in self.layers:
			out = layer.forward(out)
		return out


	def gradient_descent(self,x_train,x_test):

		for 





