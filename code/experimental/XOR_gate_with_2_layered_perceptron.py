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



X = np.array([[0,0],[0,1],[1,0],[1,1]])
Y = np.array([[0],[1],[1],[0]])
biases = np.ones((4,1))
X = np.hstack((X,biases))
weights_1 = np.random.randn(2,3)
weights_2 = np.random.randn(2,3)

print weights_1, relu(weights_1)

# weights = train(X,Y,weights)
# predict(X,Y,weights)



