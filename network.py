from layers import *
import numpy as np
from utils import *


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

		for layer in self.layers:







def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        print(output)
        print(target)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))






