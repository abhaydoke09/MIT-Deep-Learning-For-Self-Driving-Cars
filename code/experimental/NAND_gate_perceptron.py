import numpy as np 

def activation(val):
	return 1 if val>0.0 else 0

X = np.array([[0,0],[0,1],[1,0],[1,1]])
Y = np.array([[1],[1],[1],[0]])
biases = np.ones((4,1))
X = np.hstack((X,biases))

weights = np.random.randn(3)

print activation(0), activation(-2.0), activation(1.2),


