import numpy as np 
import math

sigmoid = lambda x: 1/(1 + np.exp(-x))
relu = lambda x: x*(x>0)
threshold = lambda x: 1*(x>0)
sigmoid = lambda x: 1/(1+np.exp(-x))
step = lambda x: 1*(x>0.5)

def activation(val):
	return 1 if val>0.0 else 0

def train(X, Y, W):
	for n in xrange(10000):
		for d in xrange(X.shape[0]):
			loss = (activation(np.dot(X[d],weights)) - Y[d])
			weights[X[d]>0] -= loss*learning_rate

	return W


def loss(Y, target):
	return 0.5*((Y - target)**2)


def predict():
	for n in range(X.shape[0]):
		#print weights_1.shape, X[0][:,np.newaxis].shape, b1.shape
		a1 = sigmoid((np.dot(weights_1, X[n][:,np.newaxis]))+b1)
		#print a1.shape
		#print weights_2.shape, a1.shape, b2.shape
		a2 = sigmoid((np.dot(weights_2, a1))+b2)

		print X[n], Y[n], a2



def backprop(X, Y, pred, weights_1, b1, a1, z1, weights_2, b2, a2, z2, l):
	#print a2.shape
	dpred = (l*2**0.5)

	da2 = dpred * a2 * (a2>0.5)	

	print 'da2 = ',da2,'\n'
	#print da2.shape					
	print 'z2 ', z2
	#dz2 = learning_rate*da2*z2*(z2>0)
	dz2 = learning_rate*da2*sigmoid(z2)*(1-sigmoid(z2))
	print 'dz2 ', dz2
	dweights_2 = learning_rate*dz2[0]*a1
	#print weights_2.shape, dweights_2.T.shape
	print 'dweights_2 = ',dweights_2
	weights_2 -= dweights_2.T
	#print weights_2

	db2 = learning_rate*dz2
	b2 -= db2

	
	#print dweights_2.shape, a1.shape
	da1 = dz2*weights_2
	print 'da1 = ', da1
 	
	dz1 = da1.T*sigmoid(z1)*(1-sigmoid(z1))
	#print da1.shape
	#print weights_1*da1*X

	#print weights_1.shape, da1.shape, X.shape
	print 'dz1 = ', dz1
	print 'X.T = ', X[:np.newaxis].T
	dweights_1 = learning_rate*dz1*X[:np.newaxis].T
	print "dweights_1 = ",dweights_1
	#print dweights_1.shape
	weights_1 -= dweights_1
	db1 = learning_rate*dz1
	b1 -= db1

	return (weights_1, b1, weights_2, b2)


X = np.array([[0,1],[0,0],[1,0],[1,1]])
Y = np.array([[1],[0],[1],[0]])


weights_1 = np.random.randn(3,2)
b1 = np.random.randn(3,1)
weights_2 = np.random.randn(1,3)
b2 = np.random.randn(1,1)
learning_rate = 0.1

for i in range(1):
	for n in range(X.shape[0]):
		print 'X = ', X[n], '\n'
		print 'weights_1 = ',weights_1, '\nb1 = ',b1, '\nweights_2 = ', weights_2, '\nb2 = ', b2
		#print weights_1.shape, X[0][:,np.newaxis].shape, b1.shape
		z1 = np.dot(weights_1, X[n][:,np.newaxis])+b1
		a1 = sigmoid(z1)
		print "a1 = ", a1
		#print weights_2.shape, a1.shape, b2.shape
		z2 = (np.dot(weights_2, a1))+b2
		a2 = sigmoid(z2)
		pred = step(a2)
		print 'a2 before threshold ',(np.dot(weights_2, a1))+b2, '\n'
		print 'a2 = ',a2
		l = loss(pred[0], Y[n])
		print 'loss =',l
		#print 'loss =',l 
		weights_1, b1, weights_2, b2 = backprop(X[n], Y[n], pred, weights_1, b1, a1, z1, weights_2, b2, a2[0], z2, l)
		print '\n'

predict()
print weights_1, b1, weights_2, b2
