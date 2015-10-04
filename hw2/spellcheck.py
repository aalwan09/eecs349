import csv
import numpy as np
import matplotlib.pyplot as plt

def levenshtein_distance(string1, string2, deletionCost, insertionCost, substitutionCost):
	
	m = len(string1)
	n = len(string2)

	
	d = [[0 for x in range(n+1)] for x in range(m+1)]
	
	for i in range(1,m):
		d[i][0] = i * deletionCost
	for j in range(1,n):
		d[0][j] = j * insertionCost

	for j in range(1,n+1):
		for i in range(1,m+1):
			if (string1[i-1] == string2[j-1]):
				d[i][j] = d[i-1][j-1]
			else:
				options = [d[i-1][j] + deletionCost, d[i][j-1]+insertionCost, d[i-1][j-1] + substitutionCost]
				d[i][j] = min(options)

	return d[m][n]

def find_closest_word(string1, dictionary):
	min_distance = 99999999
	closest_string = ""

	for tester in dictionary:
		distance = levenshtein_distance(string1, tester, 1,1,1)
		if (distance < min_distance):
			min_distance = distance
			closest_string = tester

	return tester

		
	

if __name__ == "__main__":
	print levenshtein_distance("auto bahn","fauto bahn",1,1,1)
	
	
