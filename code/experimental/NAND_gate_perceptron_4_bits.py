import numpy as np 

get_bin = lambda x, n: format(x, 'b').zfill(n)

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


def getData(nbits):
	X = []
	Y = []
	for n in range(2**nbits):
		data_point = [int(i) for i in list(get_bin(n, nbits))]
		X.append(data_point)
		Y.append([reduce(lambda x, y: x*y, data_point)])
	return np.array(X),np.array(Y)

nbits = 4
X,Y = getData(nbits)
print X.shape,Y.shape
# biases = np.ones((4,1))
# X = np.hstack((X,biases))
# weights = np.random.randn(3)
# learning_rate = 0.1

# weights = train(X,Y,weights)
# predict(X,Y,weights)



