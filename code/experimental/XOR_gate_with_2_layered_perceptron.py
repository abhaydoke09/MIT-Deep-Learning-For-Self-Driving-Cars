import numpy as np 
import math

sigmoid = lambda x: 1/(1 + np.exp(-x))
relu = lambda x: x*(x>0)

def activation(val):
	return 1 if val>0.0 else 0

def train(X, Y, W):
	for n in xrange(10000):
		for d in xrange(X.shape[0]):
			loss = (activation(np.dot(X[d],weights)) - Y[d])
			weights[X[d]>0] -= loss*learning_rate

	return W

def predict(X, Y, W):
	print '{:^12}'.format('X'), '{:^9}'.format('Predicted'), '{:^6}'.format('Target')
	for d in xrange(X.shape[0]):
		print '{:^12}'.format(X[d]), '{:^9}'.format(activation(np.dot(X[d],W))), '{:^6}'.format(Y[d])

def loss(Y, target):
	return (Y - target)**2




X = np.array([[0,0],[0,1],[1,0],[1,1]])
Y = np.array([[0],[1],[1],[0]])


weights_1 = np.random.randn(3,2)
b1 = np.random.randn(3)
weights_2 = np.random.randn(1,3)
b2 = np.random.randn(1)

for n in range(X.shape[0]):
	a1 = relu((np.dot(weights_1, X[n]))+b1)
	a2 = relu((np.dot(weights_2, a1))+b2)
	print loss(a2, Y[n])
