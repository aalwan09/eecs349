# - *- coding: utf- 8 - *-


# import modules you need here.
import sys
import numpy as np

import scipy
import scipy.stats 

import random

import item_cf


class errorMeasure():
	def __init__(self,k,i,distance,MPE, MSE):
		self.k = k
		self.i = i
		self.distance = distance
		self.MPE = MPE[:]
		self.MSE = MSE[:]
		
		
results_user = []
results_item = []


def main():
	datafile = sys.argv[1]
	
	
	##create a sample set consisting of 50 samples that each have 100 reviews
	##put together the rest as the training set
	##write both to files
	
	##generate a testing set of size 100 and a training set of 99 900 
	##each example in the testing set needs to have at least 100 reviews
	counter = 0
	inputArray = np.loadtxt(datafile)
	k_set = [1,2,4,8,16,32]
	i_set = [0,1]
	distance_set = [0,1]
	trials = 50
	N = 100
	
	##for each setting run this trials times and after trials times create an errorMeasure with the settings and two arrays corresponding to the individual error measures MPE, MSE
	
	for k in k_set:
		for i in i_set:
			for distance in distance_set:
				##inside parameter for loops:
				MSE_item = []
				MPE_item = []
				MSE_user = []
				MPE_user = []
				print "doing " + str(trials) + " runs with parameters: k =" + str(k) + " distance =" +str(distance) + " i =" + str(i)
				print "did " + str(counter) + " calculations so far"
				for j in range(0,trials):	
					##inside trials times loop:
					
					testSamples = []
					allData = inputArray[:]
					random_indexes = random.sample(range(0,len(inputArray)-1), N)
					random_indexes.sort(reverse=True)
					for i in random_indexes:
						testSamples.append(allData[i])
						allData = np.delete(allData, i,0)
						
					maxUser = 0;
					maxMovie = 0;
					for row in allData:
						if row[0] > maxUser:
							maxUser = row[0]	
						if row[1] > maxMovie:
							maxMovie = row[1]
							
					dataArray = np.zeros((int(maxUser), int(maxMovie)))
					
					for row in allData:
						dataArray[int(row[0])-1][int(row[1])-1] = row[2]
				
					MSE_local = 0
					MPE_local = 0
					for sample in testSamples:
						##do test 
						##if result = true, then don't update MPE_local
						##definitely update MSE_local
						counter += 1
						
				##create an object and append it to the global results array
				##this is after the trials times loop
				results_user.append(errorMeasure(k,i,distance,MPE_user,MSE_user))
				results_item.append(errorMeasure(k,i,distance,MPE_item,MSE_item))
	print counter
	f_user = open('user_cf_results.txt','w')
	f_user.write(results_user)
	f_user.close()
	f_item = open('item_cf_results.txt','w')
	f_item.write(results_item)
	f_item.close()



if __name__ == "__main__":
	main()
