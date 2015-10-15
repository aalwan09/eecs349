import sys
import csv
import numpy as np
#import scipy
import matplotlib.pyplot as plt


def perceptrona(w_init, X, Y):
	#figure out (w, k) and return them here. w is the vector of weights, k is how many iterations it took to converge.
	print w_init, X, Y
	k = 0
	w = w_init
	solution = False
	total_count = len(Y)
	while solution is not True:
		correct_count = 0
		#print "new try"
		for xk,yk in zip(X,Y):
			k += 1
			g_x = xk*w[0]+w[1]
			h_x = 0
			if (g_x > 0):
				h_x = 1
			else:
				h_x = -1
			#print "Classification says: " + str(h_x) + " is supposed to be: " + str(yk)
			if (h_x == yk):
				correct_count += 1
			else:
				w = w + yk * np.array([xk, yk])
				#print "new w: "
				#print w
				vals = np.arange(X.min(), X.max(), (X.max()+abs(X.min()))/100)
				tests = np.zeros(len(vals))
				for t in tests:
					t =w[0]+ t * w[1]
				plt.plot(X,Y,'ro', vals, tests, 'k')
				plt.show()
				break

		if (correct_count == total_count):
			solution = True
			print "all good!"

	return (w, k)


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
	
	w_init = np.zeros(2)# INTIALIZE W_INIT
	w, k = perceptrona(w_init, X1, Y1)
	vals = np.arange(X1.min(), X1.max(), (X1.max()+abs(X1.min()))/100)
	tests = np.zeros(len(vals))
	for t in tests:
		t =w[0]+ t * w[1]
	print "Convergence took " + str(k) + " trials"
	plt.plot(X1,Y1,'ro', vals, tests, 'k')
	plt.show()
	
	w_init = np.zeros(2)# INTIALIZE W_INIT
	w, k = perceptrona(w_init, X2, Y2)
	vals = np.arange(X2.min(), X2.max(), (X2.max()+abs(X2.min()))/100)
	tests = np.zeros(len(vals))
	for t in tests:
		t =w[0]+ t * w[1]
	print "Convergence took " + str(k) + " trials"
	plt.plot(X2,Y2,'ro', vals, tests, 'k')
	plt.show()
	
	

if __name__ == "__main__":
	main()
