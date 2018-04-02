import numpy as np


def h(X,y):
	return X.dot(y)


def computeCost(X, y, theta):
	"""
	   computes the cost of using theta as the parameter for linear
	   regression to fit the data points in X and y
	"""
	m = y.size
	J = 0

# ====================== YOUR CODE HERE ======================
# Instructions: Compute the cost of a particular choice of theta
#			   You should set J to the cost.

	J = np.sum(pow(X.dot(theta) - y, 2)) / (2.0*m)

# =========================================================================

	return J


