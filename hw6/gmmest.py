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
