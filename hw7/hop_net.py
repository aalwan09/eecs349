import numpy as np
import csv
import math as m
from matplotlib import pyplot as plt

def display_vector_as_image(image):
	plt.imshow(np.reshape(image, (16, 16)), interpolation="nearest")
	plt.show()

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
	s = np.copy(x)
	number_nodes = len(x)
	#print number_nodes
	rand_index = range(number_nodes)
	np.random.shuffle(rand_index)
	#print rand_index
	iteration = 0
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
		iteration += 1
	print "took " + str(iteration) + " iterations"
	return s

def parse_data(f):
	csvfile = open(f, 'rb')
	data = csv.reader(csvfile, delimiter = ' ')
	out = []
	label_extract = [0,1,2,3,4,5,6,7,8,9]
	for row in data:
		local_row = np.zeros(len(row[:-11]))
		for i,d in enumerate(row[:-11]):
			local_row[i]= int(float(d))
			if local_row[i] == 0:
				local_row[i]= -1
		label = row[-11:]
		label = label[:10]
			
		for i,d in enumerate(label):
			label[i] = int(float(label[i]))
		label_name = np.dot(label, label_extract)
		out.append((label_name, local_row))
	return out

def probc(X):
	training_set = []
	testing_set = []
	indices_training = []
	l = 0
	for i in range(len(X)):
		if X[i][0] == l:
			training_set.append(X[i][1])
			l+= 1
			indices_training.append(i)
		if l == 5:
			break

	for i in training_set:
		display_vector_as_image(i)
	
	l = 3
	for i in range(len(X)):
		if X[i][0] == l and i not in indices_training:
			testing_set = X[i][1]
			break

	W = train_hopfield(np.array(training_set))
	test = use_hopfield(W, testing_set)

	

def main():
	X = parse_data("semeion.data")
	probc(X)
		

if __name__ == "__main__":
	main()

