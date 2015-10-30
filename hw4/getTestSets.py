# - *- coding: utf- 8 - *-


# import modules you need here.
import sys
import numpy as np
import math
import scipy
import scipy.stats 

import random


#optimization stuff
#from numba import autojit

class errorMeasure():
	def __init__(self,k,i,distance,MPE, MSE):
		self.k = k
		self.i = i
		self.distance = distance
		self.MPE = MPE[:]
		self.MSE = MSE[:]
		
		
results_user = []
results_item = []


def manhattan_dist(a,b):
	distance = 0
	for i,j in enumerate(a):
		distance += abs(j-b[i])
	return distance


def user_based_cf(datafile, userid, movieid, distance, k, iFlag):
	'''
	build user-based collaborative filter that predicts the rating 
	of a user for a movie.
	This function returns the predicted rating and its actual rating.

	Parameters
	----------
	<datafile> - a fully specified path to a file formatted like the MovieLens100K data file u.data 
	<userid> - a userId in the MovieLens100K data
	<movieid> - a movieID in the MovieLens 100K data set
	<distance> - a Boolean. If set to 0, use Pearson’s correlation as the distance measure. If 1, use Manhattan distance.
	<k> - The number of nearest neighbors to consider
	<iFlag> - A Boolean value. If set to 0 for user-based collaborative filtering, 
	only users that have actual (ie non-0) ratings for the movie are considered in your top K. 
	For user-based, use only movies that have actual ratings by the user in your top K. 
	If set to 1, simply use the top K regardless of whether the top K contain actual or filled-in ratings.

	returns
	-------
	trueRating: <userid>'s actual rating for <movieid>
	predictedRating: <userid>'s rating predicted by collaborative filter for <movieid>


	AUTHOR: Marc Gyongyosi (This is where you put your name)
	'''
	if type(datafile) is str:
  		inputArray = np.loadtxt(datafile)
  	else:
  		inputArray = datafile[:]
  	if ((distance != 0)&(distance!=1)):
  		print "error!!!! FUCK!"
  		print distance
  	if ((iFlag != 0)&(iFlag!=1)):
  		print "error!!!! FUCK!"
  		print iFlag
	maxUser = 0;
	maxMovie = 0;
	for row in inputArray:
		if row[0] > maxUser:
			maxUser = row[0]	
		if row[1] > maxMovie:
			maxMovie = row[1]
	dataArray = np.zeros((int(maxUser), int(maxMovie)))
	for row in inputArray:
		dataArray[int(row[0])-1][int(row[1])-1] = row[2]

	distancesArr = []
	
	target_user = dataArray[userid-1]

	if (iFlag == 1):
		for i, j in enumerate(dataArray):
			if (i == (userid-1)):
				distancesArr.append(-99999);
			else:
				if (distance == 0):
					distancesArr.append(scipy.stats.pearsonr(target_user, j)[0])
				elif (distance == 1):
					distancesArr.append(-1*manhattan_dist(target_user, j))

		#k_indexes_of_min_distance = np.argpartition(distancesArr, -k)[-k:]
		k_indexes_of_min_distance = sorted(range(len(distancesArr)), key=lambda x: distancesArr[x])[-k:]
		k_ratings = [];
		for index in k_indexes_of_min_distance:
			k_ratings.append(dataArray[index][movieid-1])
		predictedRating = scipy.stats.mode(k_ratings)[0][0]
		#print "k_ratings:"
		#print k_ratings
		trueRating = dataArray[userid-1][movieid-1]

	if (iFlag == 0):
		for i, j in enumerate(dataArray):
			if (i == (userid-1)):
				distancesArr.append(-99999);
			elif (j[movieid-1] == 0.):
				distancesArr.append(-99999);
			else:
				if (distance == 0):
					distancesArr.append(scipy.stats.pearsonr(target_user, j)[0])
				elif (distance == 1):
					distancesArr.append(-1*manhattan_dist(target_user, j))

		#k_indexes_of_min_distance = np.argpartition(distancesArr, -k)[-k:]
		k_indexes_of_min_distance = sorted(range(len(distancesArr)), key=lambda x: distancesArr[x])[-k:]
		k_ratings = [];
		for index in k_indexes_of_min_distance:
			k_ratings.append(dataArray[index][movieid-1])
		predictedRating = scipy.stats.mode(k_ratings)[0][0]

		trueRating = dataArray[userid-1][movieid-1]			
		

	return trueRating, predictedRating

def item_based_cf(datafile, userid, movieid, distance, k, iFlag):
	'''
	build item-based collaborative filter that predicts the rating 
	of a user for a movie.
	This function returns the predicted rating and its actual rating.
	Parameters
	----------
	<datafile> - a fully specified path to a file formatted like the MovieLens100K data file u.data 
	<userid> - a userId in the MovieLens100K data
	<movieid> - a movieID in the MovieLens 100K data set
	<distance> - a Boolean. If set to 0, use Pearson’s correlation as the distance measure. If 1, use Manhattan distance.
	<k> - The number of nearest neighbors to consider
	<iFlag> - A Boolean value. If set to 0 for user-based collaborative filtering, 
	only users that have actual (ie non-0) ratings for the movie are considered in your top K. 
	For item-based, use only movies that have actual ratings by the user in your top K. 
	If set to 1, simply use the top K regardless of whether the top K contain actual or filled-in ratings.
	returns
	-------
	trueRating: <userid>'s actual rating for <movieid>
	predictedRating: <userid>'s rating predicted by collaborative filter for <movieid>
	AUTHOR: Marc Gyongyosi
	'''
	
	if type(datafile) is str:
  		inputArray = np.loadtxt(datafile)
  	else:
  		inputArray = datafile[:]
  		
  	if ((distance != 0)&(distance!=1)):
  		print "error distance!!!!"
  		print distance
  	if ((iFlag != 0)&(iFlag!=1)):
  		print "error iFlag!!!!"
  		print iFlag
	maxUser = 0;
	maxMovie = 0;
	for row in inputArray:
		if row[0] > maxUser:
			maxUser = row[0]	
		if row[1] > maxMovie:
			maxMovie = row[1]
	dataArray = np.zeros((int(maxUser), int(maxMovie)))
	for row in inputArray:
		dataArray[int(row[0])-1][int(row[1])-1] = row[2]

	rotateArr = dataArray.view()
	rotateArr = rotateArr.T
	distancesArr = []
	target_movie = rotateArr[movieid-1]
	#print "data Array:"
	#print dataArray
	#print "rotate Array:"
	#print rotateArr	
	#print "target_movie:"
	#print target_movie
	
	if (iFlag == 1):
		for i, j in enumerate(rotateArr):
			if (i == (movieid-1)):
				distancesArr.append(-99999);
			else:
				if (distance == 0):
					distancesArr.append(scipy.stats.pearsonr(target_movie, j)[0])
				elif (distance == 1):
					distancesArr.append(-1*manhattan_dist(target_movie, j))
		#print "distancesArr:"
		#print distancesArr
		#k_indexes_of_min_distance = np.argpartition(distancesArr, -k)[-k:]
		k_indexes_of_min_distance = sorted(range(len(distancesArr)), key=lambda x: distancesArr[x])[-k:]
		#print "k_indexes:"
		#print k_indexes_of_min_distance

		k_ratings = [];
		for index in k_indexes_of_min_distance:
			k_ratings.append(dataArray[userid-1][index])
		#print "k_ratings:"
		#print k_ratings
		predictedRating = scipy.stats.mode(k_ratings)[0][0]
		
		trueRating = dataArray[userid-1][movieid-1]


	if (iFlag == 0):
		for i, j in enumerate(rotateArr):
			if (i == (movieid-1)):
				distancesArr.append(-99999);
			elif (j[userid-1] == 0.):
				distancesArr.append(-99999);
			else:
				if (distance == 0):
					distancesArr.append(scipy.stats.pearsonr(target_movie, j)[0])
				elif (distance == 1):
					distancesArr.append(-1*manhattan_dist(target_movie, j))

		#k_indexes_of_min_distance = np.argpartition(distancesArr, -k)[-k:]
		k_indexes_of_min_distance = sorted(range(len(distancesArr)), key=lambda x: distancesArr[x])[-k:]
		k_ratings = [];
		for index in k_indexes_of_min_distance:
			k_ratings.append(dataArray[userid-1][index])
		#print "k_ratings:"
		#print k_ratings
		predictedRating = scipy.stats.mode(k_ratings)[0][0]

		trueRating = dataArray[userid-1][movieid-1]			
		

	return trueRating, predictedRating



def main():
	datafile = sys.argv[1]
	
	
	##create a sample set consisting of 50 samples that each have 100 reviews
	##put together the rest as the training set
	##write both to files
	
	##generate a testing set of size 100 and a training set of 99 900 
	##each example in the testing set needs to have at least 100 reviews
	counter = 0
	inputArray = np.loadtxt(datafile)
	#k_set = [1,2,4,8,16,32]
	k_set = [16]
	i_set = [0,1]
	i_set = [0]
	distance_set = [0,1]
	distance_set = [0]
	trials = 5
	N = 10
	
	##for each setting run this trials times and after trials times create an errorMeasure with the settings and two arrays corresponding to the individual error measures MPE, MSE
	
	for k in k_set:
		for i_f in i_set:
			for distance in distance_set:
				##inside parameter for loops:
				
				MSE_item = []
				MPE_item = []
				MSE_user = []
				MPE_user = []
				print "did " + str(counter) + " calculations so far"
				for j in range(0,trials):	
					##inside trials times loop:
					
					testSamples = []
					trainingData = inputArray[:]
					random_indexes = random.sample(range(0,len(inputArray)-1), N)
					random_indexes.sort(reverse=True)
					for i_r in random_indexes:
						testSamples.append(trainingData[i_r])
						trainingData = np.delete(trainingData, i_r,0)
				
					MSE_local_item = 0
					MPE_local_item = 0
					MSE_local_user = 0
					MPE_local_user = 0
					print len(testSamples)
					
					for sample in testSamples:
						##do test 
						##if result = true, then don't update MPE_local
						##definitely update MSE_local
						userid = sample[0]
						movieid = sample[1]
						sample_true = sample[2]
						print "_________________________________"
						print "_________________________________"
						print "Test# " +str(counter+1)
						print "_________________________________"
						print "userid: " + str(userid) + " movieid: " + str(movieid) + " distance: " +str(distance) + " k: " + str(k) + " i: " +str(i_f)
						item_trueRating, item_predictedRating = item_based_cf(trainingData, userid, movieid, distance, k, i_f)
						user_trueRating, user_predictedRating = user_based_cf(trainingData, userid, movieid, distance, k, i_f)
						
						print "Results: \tTRUE RATING: " + str(sample_true) 
						print "         \tITEM_BASED: " + str(item_predictedRating) + "\tUSER_BASED: " +str(user_predictedRating)
						
						##item based eval
						
						if (sample_true != item_predictedRating):
							MPE_local_item += 1
						MSE_local_item += math.pow((sample_true - item_predictedRating),2)
						
						##user based eval
						if (sample_true != user_predictedRating):
							MPE_local_user += 1
						MSE_local_user += math.pow((sample_true - user_predictedRating), 2)
						
						counter += 1
						
						
						
					MSE_local_item = float(MSE_local_item) / float(N)
					MPE_local_item = float(MPE_local_item) / float(N)
					MSE_local_user = float(MSE_local_user) / float(N)
					MPE_local_user = float(MPE_local_user) / float(N)
					MSE_item.append(MSE_local_item)
					MPE_item.append(MPE_local_item)
					MSE_user.append(MSE_local_user)
					MPE_user.append(MPE_local_user)		
					print "MSE_local_item " + str(MSE_local_item)
					print "MPE_local_item " + str(MPE_local_item)
					print "MSE_local_user " + str(MSE_local_user)
					print "MPE_local_user " + str(MPE_local_user)
				##create an object and append it to the global results array
				##this is after the trials times loop
				results_user.append(errorMeasure(k,i_f,distance,MPE_user,MSE_user))
				results_item.append(errorMeasure(k,i_f,distance,MPE_item,MSE_item))
				print counter
				
	results_user_string = "k\ti\tdistance\tMPE\tMSE \n"
	results_item_string = "k\ti\tdistance\tMPE\tMSE \n"
	
	for result in results_user:
		results_user_string += (str(result.k) + "\t" + str(result.i) + "\t" + str(result.distance) + "\t" + str(result.MPE) + "\t" + str(result.MSE) ) + "\n"
	
	for result in results_item:
		results_item_string += (str(result.k) + "\t" + str(result.i) + "\t" + str(result.distance) + "\t" + str(result.MPE) + "\t" + str(result.MSE) ) + "\n"
	
	f_user = open('user_cf_results.txt','w')
	f_user.write(results_user_string)
	f_user.close()
	f_item = open('item_cf_results.txt','w')
	f_item.write(results_item_string)
	f_item.close()



if __name__ == "__main__":
	main()
