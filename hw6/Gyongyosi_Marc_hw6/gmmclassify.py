import sys
import time
import numpy as np
import scipy.stats 
import csv
import math
import matplotlib.pyplot as plt

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
