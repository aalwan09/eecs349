import sys
import time
import numpy as np
import scipy.stats 
import csv
import math
import matplotlib.pyplot as plt

def gmmest(X,mu_init,sigmasq_init,wt_init,its):
	# Use numpy vectors/arrays.
	# Input
	# - X		: N 1-dimensional data points (a 1-by-N vector)
	# - mu_init	: initial means of K Gaussian components
	#			(a 1-by-K vector)
	# - sigmasq_init: initial variances of K Gaussian components
	# 			(a 1-by-K vector)
	# - wt_init	: initial weights of k Gaussian components
	#			(a 1-by-K vector that sums to 1)
	# - its		: number of iterations for the EM algorithm
	#
	# Output
	# - mu		: means of Gaussian components (a 1-by-K vector)
	# - sigmasq	: variances of Gaussian components (a 1-by-K vector)
	# - wt		: weights of Gaussian components (a 1-by-K vector, sums
	#			to 1)
	# - L		: log likelihood
	k = len(mu_init)	
	L = []

	N = len(X)

	for i in range(its):
		for j in range (k):
			resp_xn = []
			resp_total = 0

			##E STEP
			for xn in X:
				prob = scipy.stats.norm(mu_init[j], np.sqrt(sigmasq_init[j])).pdf(xn)
				temp = 0
				for k_local in range(k):
					temp += wt_init[k_local] * scipy.stats.norm(mu_init[k_local], np.sqrt(sigmasq_init[k_local])).pdf(xn)

				if temp == 0:
					resp_xn.append(0.)
				else:
					resp_xn.append(wt_init[j]*prob/temp)

			for r in resp_xn:
				resp_total += r

			##M STEP
			wt_init[j] = resp_total/N


			#temp2 = 0
			#for i in range(N):
			#	temp2 += resp_xn[i] * X[i]
			#mu_init[j] = temp2/resp_total

			mu_init[j] = np.dot(resp_xn, X)/resp_total
			
			temp2 = []
			for x in X:
				temp2.append((x-mu_init[j])**2)
			sigmasq_init[j] = np.dot(resp_xn, temp2) / resp_total

		L_local = 0
		for x in X:
			temp3 = 0
			for k_local in range(k):
				temp3 += wt_init[k_local] * scipy.stats.norm(mu_init[k_local], np.sqrt(sigmasq_init[k_local])).pdf(x)
			L_local += np.log(temp3)

		L.append(L_local)

	return mu_init, sigmasq_init, wt_init, L
					

def gmmclassify(X,mu1,sigmasq1,wt1,mu2,sigmasq2,wt2,p1):
	output = np.zeros(len(X))

	for idx, x in enumerate(X):
		p1 = -1;
		p2 = -1;
		
		for i in range(len(mu1)):
			tmp = wt1[i] * scipy.stats.norm(mu1[i], np.sqrt(sigmasq1[i])).pdf(x)
			if tmp > p1:
				p1 = tmp
		for i in range(len(mu2)):
			tmp = wt2[i] * scipy.stats.norm(mu2[i], np.sqrt(sigmasq2[i])).pdf(x)
			if tmp > p2:
				p2 = tmp

		if p1>p2:
			output[idx] = int(1)
		else:
			output[idx] = int(2)
			
	
	return output

def main():
	test_file = sys.argv[1]
	train_file = sys.argv[2]

	f = open(test_file, 'rb')
	data = csv.reader(f, delimiter=',')
	X_test = []
	Y_test = []

	for row in data:
		try:
			X_test.append(float(row[0]))
			Y_test.append(int(row[1]))
		except:
			print "Could not parse data:"
			print row

	f.close()
	X_test = np.array(X_test)
	Y_test = np.array(Y_test)

	f = open(train_file, 'rb')
	data = csv.reader(f, delimiter=',')
	X_train = []
	Y_train = []

	for row in data:
		try:
			X_train.append(float(row[0]))
			Y_train.append(int(row[1]))
		except:
			print "Could not parse data:"
			print row
	f.close()
	X_train = np.array(X_train)
	Y_train = np.array(Y_train)

	##VISUALIZATION:
	class1 = X_train[np.nonzero(Y_train ==1)[0]]
	class2 = X_train[np.nonzero(Y_train ==2)[0]]
	bins = 50 # the number 50 is just an example.
	plt.hist(class1, bins, label='Class1', alpha=0.5)
	plt.hist(class2, bins, label='Class2', alpha=0.5)
	plt.xlabel('value of datapoint')
	plt.ylabel('number of recurrences w.r.t class')
	plt.show()
	
	###MODEL 1:
	mu_init = [11,30] ### based on histogram plot
	sigmasq_init = [1,1]
	wt_init = [.6,.4]
	its = 15

	mu_m1, sigmasq_m1, wt_m1, L_m1 = gmmest(class1, mu_init, sigmasq_init, wt_init, its)

	##correct model:
	#mu_m1 = [9.7748859239173544, 29.582587183104856]
	#sigmasq_m1 = [21.92280456650424, 9.7837696118328275]
	#wt_m1 = [0.59765463040062106, 0.40234536960541073]
	
	print "_________________________________"
	print "             MODEL 1:            "
	print "   - MEAN:     " + str(mu_m1)
	print "   - VARIANCE: " + str(sigmasq_m1)
	print "   - WEIGHTS:  " + str(wt_m1)

	plt.plot(L_m1)
	plt.ylabel('log-likelihood of class 1')
	plt.xlabel('number of iterations')
	plt.show()

	###MODEL 2:
	mu_init = [-25,-5,48] ### based on histogram plot
	sigmasq_init = [1,1,1]
	wt_init = [.2,.4,.4]
	its = 15

	mu_m2, sigmasq_m2, wt_m2, L_m2 = gmmest(class2, mu_init, sigmasq_init, wt_init, its)

	#correct model:
	#mu_m2 =   [-24.822751728656179, -5.0601582832182022, 49.624444719527538]
	#sigmasq_m2 =  [7.9473354079952401, 23.322661814154543, 100.02433750441253]
	#wt_m2 =  [0.20364945852846328, 0.49884302379533918, 0.29750751767685873]



	print "_________________________________"
	print "             MODEL2:            "
	print "   - MEAN:     " + str(mu_m2)
	print "   - VARIANCE: " + str(sigmasq_m2)
	print "   - WEIGHTS:  " + str(wt_m2)

	plt.plot(L_m2)
	plt.ylabel('log-likelihood of class 2')
	plt.xlabel('number of iterations')
	
	plt.show()

	PP = len(class1)/float(len(X_train))
	print "_________________________________"
	print "---------------------------------"
	print "    PRIOR PROBABILITY: " + str(PP)


	cl = gmmclassify(X_test, mu_m1, sigmasq_m1, wt_m1, mu_m2, sigmasq_m2, wt_m2, PP)

	errors = 0

	for index in range(len(cl)):
		if cl[index] != Y_test[index]:
			errors += 1
	error = float(errors)/float(len(Y_test))	

	print "_________________________________"
	print "    GMM ERROR RATE: " +str(error)

	class1 = X_test[np.nonzero(Y_test==1)[0]]
	class2 = X_test[np.nonzero(Y_test==2)[0]]
	#print len(X_test)
	#print len(class1)
	#print len(class2)
	#print cl
	plt.scatter(X_test, [0.06]*len(X_test), c=cl, alpha=0.3)
	plt.hist(class1, bins, alpha =0.5, label='Class1', normed=True)
	plt.hist(class2, bins, alpha =0.5, label='Class2', normed=True)
	plt.show()


if __name__ == "__main__":
	main()
