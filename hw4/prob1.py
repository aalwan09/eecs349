import sys
import csv
import numpy as np
import matplotlib.pyplot as plt


### GPU optimization added, uncomment this line as well as all @autojit commands to leverage the GPU
from numba import autojit


@autojit
def getcommon(a,b):
	common = 0
	for idx, val in enumerate(a):
		if ((val != 0) & (b[idx] != 0)):
			common += 1
	return common

@autojit
def countreviews(a):
	reviews = 0
	for user in a:
		if (user != 0):
			reviews += 1
	return reviews

##quicksort algorithm, python implementation from http://rosettacode.org/wiki/Sorting_algorithms/Quicksort#Python
def qsort(inlist):
	if inlist == []: 
		return []
	else:
		pivot = inlist[0]
		lesser = qsort([x for x in inlist[1:] if x > pivot])
		greater = qsort([x for x in inlist[1:] if x <= pivot])
	return lesser + [pivot] + greater
	
@autojit
def main():
	dataArray = np.zeros((943, 1682))
	inputArray = np.loadtxt('ml-100k/u.data')
	for row in inputArray:
		dataArray[int(row[0])-1][int(row[1])-1] = row[2]
	if (sys.argv[1] == 'a'):
		pair_common = []
		first_index = 1
		count = 0
		for user in dataArray:
			for user2 in dataArray[first_index:len(dataArray)]:
				if (first_index < len(dataArray)):
					pair_common.append(getcommon(user, user2))
					count += 1
					#print "did pair #" + str(count) + " out of 444153"
			first_index += 1
		pair_common_sorted = []
		print max(pair_common)
	
		for i in range(0, max(pair_common)+1):
			pair_common_sorted.append( pair_common.count(i))
		print pair_common_sorted
		#pair_common_sorted = pair_common_sorted[0:100]

		fig = plt.figure()
		ax = fig.add_subplot(111)
		width = 0.5
		rects1 = ax.bar(range(0,len(pair_common_sorted)), pair_common_sorted, width, color='blue')
		ax.set_xlabel('number of reviews in common')
		ax.set_ylabel('number of pairs')
		ax.set_title('number of reviews vs number of pairs, histogram (complete)')
		print "mean of number of reviews in common: " + str(np.mean(pair_common))
		print "median of number of reviews in common: " + str(np.median(pair_common))
		plt.show()	

	if (sys.argv[1] == 'b'):
		dataArray_transpose = dataArray.view()
		dataArray_transpose.T
		review_counts = []
		for movie in dataArray_transpose:
			review_counts.append(countreviews(movie))

		m = max(review_counts)
		index = [] 
		for i, j in enumerate(review_counts):
			if j == m: 
				index.append(i)
		print "Maximum review count: " + str(m) + " found at index: "
		print index
		
		m = min(review_counts)
		index = [] 
		for i, j in enumerate(review_counts):
			if j == m: 
				index.append(i)
		print "Minimum review count: " + str(m) + " found at index: "
		print index

		review_counts_sorted = qsort(review_counts)
		plt.plot(range(0,len(review_counts_sorted)), review_counts_sorted)
		plt.xlabel('movie')
		plt.ylabel('number of reviews')
		plt.show()
		
		

if __name__ == "__main__":
	main()
