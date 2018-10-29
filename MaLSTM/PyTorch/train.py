import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 


def get_accuracy(net,q1,q2,labels,batchsize):
	correct = 0
	total = 0

	numbatches = int(len(q1)/batchsize)

	for i in range(numbatches):
		x1 = q1[i*batchsize:(i+1)*batchsize,:]
		x2 = q2[i*batchsize:(i+1)*batchsize,:]

		y = labels[i*batchsize:(i+1)*batchsize]

		preds = net(x1,x2)
		preds = preds.view(preds.size(0))
		ypred = (preds>=0.5).double()

		y = y.double()

		correct+=torch.sum(y==ypred).item()

		total+=x1.size(0)

	return ((correct/total)*100)



def train_model(model,train_ind1,train_ind2,valid_ind1,valid_ind2,
	test_ind1,test_ind2,train_labels,valid_labels,test_labels,batchsize,numepochs):


	optimizer = torch.optim.Adadelta(model.parameters(),weight_decay=1.25)

	numbatches = int(len(train_ind1)/batchsize)

	for epoch in range(numepochs):
		for i in range(numbatches):
			train_q1 = train_ind1[i*batchsize:(i+1)*batchsize,:]
			train_q2 = train_ind2[i*batchsize:(i+1)*batchsize,:]

			ytrain = train_labels[i*batchsize:(i+1)*batchsize]

			ytrain = ytrain.view(-1,1).float()

			ypred = model(train_q1,train_q2)

			ypred = ypred.float()

			loss = F.binary_cross_entropy(ypred,ytrain)

			loss.backward()

			optimizer.step()

		train_acc = get_accuracy(model,train_ind1,train_ind2,train_labels,batchsize)

		print("Train Loss {} and Train Accuracy {}".format(loss,train_acc))

		valid_acc = get_accuracy(model,valid_ind1,valid_ind2,valid_labels,batchsize)

		print("Validation Loss {} and Validation Accuracy {}".format(loss,valid_acc))

	test_acc = get_accuracy(model,test_ind1,test_ind2,test_labels,batchsize)

	print("Test Loss {} and Test Accuracy {}".format(loss,test_acc))


