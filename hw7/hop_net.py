import numpy as np
import csv
import math as m

def train_hopfield(X):
#train_hopfield(X)
#
# Creates a set of weights between nodes in a Hopfield network whose
# size is based on the length of the rows in the input data X.
#
# X is a numpy.array of shape (R, C). Values in X are drawn from
# {+1,-1}. Each row is a single training example. So X(3,:) would be
# the 3rd training example.
#
# W is a C by C numpy array of weights, where W(a,b) is the connection
# weight between nodes a and b in the Hopfield net after training.
# Here, C = number of nodes in the net = number of columns in X
#
# 
	R = len(X)
	C = len(X[0])
	W = np.zeros((C,C))

	for i in range(C):
		for j in range(C):
			if i == j:
				W[i][j] = 0
			else:
				for r_local in range(R):
					W[i][j] += X[r_local][i] * X[r_local][j]
	
	return W


def use_hopfield(W,x):
#use_hopfield(W, x)
#
# Takes a Hopfield net W and an input vector x and runs the
# Hopfield network until it converges.
#
# x, the input vector, is a numpy vector of length C (where C is
# the number of nodes in the net). This is a set of
# activation values drawn from the set {+1, -1}
#
# W is a C by C numpy array of weights, where W(a,b) is the connection
# weight between nodes a and b in the Hopfield net.
# Here, C = number of nodes in the net.
#
# s is a numpy vector of length C (number of nodes in the net)
# containing the final activation of the net. It is the result of
# giving x to the net as input and then running until convergence.
#
# 
	converged_counter = 0
	converged_bool = False
	s = np.array(x)
	number_nodes = len(x)
	rand_index = np.random.shuffle(range(len(number_nodes)))
	

	while not converged_bool:
		for index in rand_index:
			hop_val = -1
			if (np.dot(s, W[index]) >=0):
				hop_val = 1
			if (hop_val != s[index]):
				s[index] = hop_val
				converged_counter = 0
			else: 
				converged_counter += 1
		if converged_counter >= number_nodes:
			converged_bool = True
		
	return s


def main():
	

if __name__ == "__main__":
	main()

