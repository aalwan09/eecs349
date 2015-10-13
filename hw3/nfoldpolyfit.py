#	Starter code for linear regression problem
#	Below are all the modules that you'll need to have working to complete this problem
#	Some helpful functions: np.polyfit, scipy.polyval, zip, np.random.shuffle, np.argmin, np.sum, plt.boxplot, plt.subplot, plt.figure, plt.title
import sys
import csv
import numpy as np
import scipy
import math
import matplotlib.pyplot as plt


def evaluateX(x, W):
	out = 0
	index = 0
	for w in W:
		out = out + math.pow(float(x), index) * float(w)
		index += 1
	return out


def calcW(X_in,Y_in,m):
# 	calcW takes in two data vectors and the degree of the expected polynomials and returns the coefficients in the matrix W
	index = 0
	X = np.zeros(shape = (len(X_in), m+1))
	for x in X:
		x[0] = 1
		for i in range(1,m+1):
			x[i] = math.pow(X_in[index],i)
		index += 1
	X_T = np.matrix.transpose(X)
	
	W = np.linalg.inv(np.mat(X_T) * np.mat(X)) * X_T *np.matrix.transpose(np.mat(Y_in))
	return W

def nfoldpolyfit(X, Y, maxK, n, verbose):
#	NFOLDPOLYFIT Fit polynomial of the best degree to data.
#   NFOLDPOLYFIT(X,Y,maxDegree, nFold, verbose) finds and returns the coefficients 
#   of a polynomial P(X) of a degree between 1 and N that fits the data Y 
#   best in a least-squares sense, averaged over nFold trials of cross validation.
#
#   P is a vector (in numpy) of length N+1 containing the polynomial coefficients in
#   descending powers, P(1)*X^N + P(2)*X^(N-1) +...+ P(N)*X + P(N+1). use
#   numpy.polyval(P,Z) for some vector of input Z to see the output.
#
#   X and Y are vectors of datapoints specifying  input (X) and output (Y)
#   of the function to be learned. Class support for inputs X,Y: 
#   float, double, single
#
#   maxDegree is the highest degree polynomial to be tried. For example, if
#   maxDegree = 3, then polynomials of degree 0, 1, 2, 3 would be tried.
#
#   nFold sets the number of folds in nfold cross validation when finding
#   the best polynomial. Data is split into n parts and the polynomial is run n
#   times for each degree: testing on 1/n data points and training on the
#   rest.
#
#   verbose, if set to 1 shows mean squared error as a function of the 
#   degrees of the polynomial on one plot, and displays the fit of the best
#   polynomial to the data in a second plot.
#   
#
#   AUTHOR: Marc Gyongyosi (excl. the comments of this function)
#

	vals = np.arange(X.min(), X.max(), (X.max()+abs(X.min()))/100)
	MSE_arr = [0 for x in range(0,maxK+1)]
	MSE_min = 1000
	k_min = 0
	w_min = None
	
	for k in range(0,maxK+1):
		MSE = 0
		X_foldsets = np.split(X, n)
		Y_foldsets = np.split(Y, n)
		
		for exp in range(1, n+1):
			X_trainingset = X_foldsets[exp-1]
			Y_trainingset = Y_foldsets[exp-1]
			
			index = 0
			X_testingset = []
			Y_testingset = []
			for subset in X_foldsets:
				if (index == exp-1):
					pass
				else:
					X_testingset += subset.tolist()
					Y_testingset += subset.tolist()
				index +=1
			#print X_testingset
			#print X_trainingset
			#print X
			W = calcW(X_trainingset,Y_trainingset,k)
			MSE_local = 0
			
			for x,y in zip(X_testingset,Y_testingset):
				#print evaluateX(x,W)
				MSE_local += math.pow((y-evaluateX(x,W)), 2)
			MSE_local = MSE_local/len(X)
			
			if (MSE_local < MSE_min):
				MSE_min = MSE_local
				k_min = k
				w_min = W
			
			h = []
			for val in vals:
				h.append(evaluateX(val, W))	
			plt.plot(X,Y, 'ro', vals, h, 'k')
			plt.show()
			
			print "Polynomial: " + str(k) + " Experiment: " + str(exp) + " MSE: " + str(MSE_local)
			MSE += MSE_local
		MSE_arr[k] = MSE/n
	
	print "--------------"
	print "MSE for k in range 0 to " + str(maxK)
	print MSE_arr
	
	print "Overall minimum MSE: " + str(MSE_min) + " with k = " + str(k_min)  + " and W = " 
	print w_min
	h = []
	for val in vals:
			h.append(evaluateX(val, w_min))	
	plt.plot(X,Y,'ro', vals, h, 'k')
	plt.show()
	
	
		
	
	



def main():
	# read in system arguments, first the csv file, max degree fit, number of folds, verbose
	rfile = sys.argv[1]
	maxK = int(sys.argv[2])
	nFolds = int(sys.argv[3])
	verbose = bool(sys.argv[4])
	
	csvfile = open(rfile, 'rb')
	dat = csv.reader(csvfile, delimiter=',')
	X = []
	Y = []
	# put the x coordinates in the list X, the y coordinates in the list Y
	for i, row in enumerate(dat):
		if i > 0:
			X.append(float(row[0]))
			Y.append(float(row[1]))
	X = np.array(X)
	Y = np.array(Y)
	nfoldpolyfit(X, Y, maxK, nFolds, verbose)

if __name__ == "__main__":
	main()
