import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
import math

def perceptronc(w_init, X, Y):
#	PERCEPTRONC Find weights for linear discrimination problem with transformation onto different
#	space to enable functionality with X2 dataset.
#   PERCEPTRONC(w_init, X,Y) finds and returns the weights w as well as e, the number of 
#	epochs it took to reach convergence to solve the linear discrimination problem described by
#	samples in X with corresponding labels in Y
# 
#   w_init is a vector (in numpy) of length 2 containing the initial guess for the weights of the 
#	linear discriminants
#
#   X and Y are vectors of datapoints specifying  input (X) and output (Y)
#   of the classification to be learned. Class support for inputs X,Y: 
#   float, double, single
#
#   AUTHOR: Marc Gyongyosi 
	e = 0
	w = w_init
	solution = False
	total_count = len(Y)
	error_count = total_count
	while solution is not True:
		error_count = total_count
	
		for xk,yk in zip(X,Y):
			g_x = w[0]+w[1]*xk+ w[2] * math.pow(xk,2)
			h_x = 0
			
			if (g_x > 0):
				h_x = 1
			else:
				h_x = -1
				
			if (h_x == yk):
				error_count -= 1
			else:
				w = w + yk * np.array([1, xk, math.pow(xk,2)])
				
				
		e += 1
		if (error_count == 0):
			solution = True
			print "Done!"
		

	return (w, e)


def main():
	rfile = sys.argv[1]
	
	#read in csv file into np.arrays X1, X2, Y1, Y2
	csvfile = open(rfile, 'rb')
	dat = csv.reader(csvfile, delimiter=',')
	X1 = []
	Y1 = []
	X2 = []
	Y2 = []
	for i, row in enumerate(dat):
		if i > 0:
			X1.append(float(row[0]))
			X2.append(float(row[1]))
			Y1.append(float(row[2]))
			Y2.append(float(row[3]))
	X1 = np.array(X1)
	X2 = np.array(X2)
	Y1 = np.array(Y1)
	Y2 = np.array(Y2)
	"""
	print "--------"
	print "Starting Test 1"
	w_init = np.array([0,0])# INTIALIZE W_INIT
	w, k = perceptronc(w_init, X1, Y1)
	vals = np.arange(X1.min(), X1.max(), (X1.max()+abs(X1.min()))/100)
	tests = np.zeros(len(vals))
	for t in tests:
		t =w[0]+ t * w[1]
	print "Convergence took " + str(k) + " epochs"
	print "Weights: w_0=" + str(w[0]) + " w_1=" + str(w[1])
	plt.plot(X1,Y1,'ro', vals, tests, 'k')
	plt.show()
	"""
	print "--------"
	print "Starting Perceptron on X2, Y2"
	w_init = np.array([0,0,0])# INTIALIZE W_INIT
	w, k = perceptronc(w_init, X2, Y2)
	vals = np.arange(X2.min(), X2.max(), (X2.max()+abs(X2.min()))/100)
	tests = np.zeros(len(vals))
	for t in tests:
		t =w[0]+ t * w[1] + w[2] * math.pow(t,2)
	print "Convergence took " + str(k) + " epochs"
	print "Weights: w_0=" + str(w[0]) + " w_1=" + str(w[1]) + " w_2=" + str(w[2])
	#plt.plot(X2,Y2,'ro', vals, tests, 'k')
	#plt.show()
	
	

if __name__ == "__main__":
	main()
