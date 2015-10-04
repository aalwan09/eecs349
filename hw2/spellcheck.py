import csv
import numpy as np
import matplotlib.pyplot as plt
import sys
import time

def levenshtein_distance(string1, string2, deletionCost, insertionCost, substitutionCost):

	m = len(string1)
	n = len(string2)

	if (m < n):
		return levenshtein_distance(string2, string1, deletionCost, insertionCost, substitutionCost)

	if (n == 0):
		return len(string1)

	

	
	d = [[0 for x in range(n+1)] for x in range(m+1)]
	
	for i in range(1,m+1):
		d[i][0] = i * deletionCost
	for j in range(1,n+1):
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
			#print "found closer distance: " + string1 + "is similar to: " + tester
			#print "distance from " + string1 + " to " + tester + "=" + str(distance)
			min_distance = distance
			closest_string = tester

	return closest_string


def measure_error(typos, truewords, dictionarywords):
	error_count = 0
	tries = 0
	index = 0
	for typo in typos:
		if (find_closest_word(typo, dictionarywords) != truewords[index]):
			error_count += 1
		else:	
			pass
		index += 1
		tries += 1
	return float(error_count)/float(tries)
		
		
	

if __name__ == "__main__":
	
	if ((len(sys.argv) < 3) | (len(sys.argv) > 3)):
		sys.exit( "Wrong number of command line arguments. \n Usage: python spellcheck.py <inputFileName> <dictionaryFile> ")
	else:
		try:
			file_in = str(sys.argv[1])
			file_dict = str(sys.argv[2])
		except:
			sys.exit("The provided arguments are malformed")
	dictionary = []
	with open(file_dict, 'rb') as csvfile:
		dictreader = csv.reader(csvfile, delimiter='\n')
		for row in dictreader:
			dictionary.append(str(row[0]))
			#print row[0]
	f_in = open(file_in, "r")
	out_file = open("corrected.txt", "w")

	input_file = f_in.read()
	
	acc = ""
	output = ""
	for i in range(0,len(input_file)):
		element = input_file[i]
		if (element.isdigit() | element.isalpha()):
			acc += element
		else:
			if (acc != ""):
				#print "trying to find fix for " + acc
				fixed_string = find_closest_word(acc, dictionary)
				#print "fix returned is " + fixed_string
				output += (fixed_string + " ")
				acc = ""
			else:
				pass
			


	out_file.write(output)

	f_in.close()
	out_file.close()
	
	complete = open("wikipediatypo.txt", "r")
	lines = complete.readlines()
	complete.close()
	typos = []
	truewords = []
	
	for line in lines:
		p = line.split()
		typos.append(str(p[0]))
		truewords.append(str(p[1]))
		

	#for i in range(0, len(typos)):
		#print str(typos[i]) + " corresponds to " + str(truewords[i])
	print typos
	print truewords
	print measure_error(typos, truewords, dictionary)
	
	
	
	
	
