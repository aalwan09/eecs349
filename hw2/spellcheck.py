import csv
import numpy as np
import matplotlib.pyplot as plt
import sys
import time
import random


insertionCost = 1
deletionCost = 1
substitutionCost = 1


def levenshtein_distance(string1, string2, deletionCost, insertionCost, substitutionCost):
	#print insertionCost
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

def qwerty_levenshtein_distance(string1, string2, deletionCost, insertionCost):
	#print insertionCost
	m = len(string1)
	n = len(string2)

	if (m < n):
		return qwerty_levenshtein_distance(string2, string1, deletionCost, insertionCost)

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
				options = [d[i-1][j] + deletionCost, d[i][j-1]+insertionCost, d[i-1][j-1] + qwerty_distance(string1[i-1], string2[j-1])]
				d[i][j] = min(options)
	

	return d[m][n]

def find(l, elem):
	for row, i in enumerate(l):
		try:
			column = i.index(elem)
		except ValueError:
			continue
		return row, column
	return -1

def qwerty_distance(char1, char2):
	char1 = str(char1).lower()
	char2 = str(char2).lower()
	keyboard = [['1','2','3','4','5','6','7','8','9','0'],['q','w','e','r','t','y','u','i','o','p'],['a','s','d','f','g','h','j','k','l'],['z','x','c','v','b','n','m']]
	try:
		c1_ir, c1_ic = find(keyboard, char1)
		c2_ir, c2_ic = find(keyboard, char2)
	except:
		print "exception in qwerty dist "
		print char1
		print char2	
	return (abs(c1_ir-c2_ir)+abs(c1_ic-c2_ic))
	
    	

def find_closest_word(string1, dictionary):
	global insertionCost
	global deletionCost 
	global substitutionCost
	min_distance = 99999999
	closest_string = ""

	for tester in dictionary:
		distance = levenshtein_distance(string1, tester, deletionCost,insertionCost,substitutionCost)
		if (distance < min_distance):
			#print "found closer distance: " + string1 + "is similar to: " + tester
			#print "distance from " + string1 + " to " + tester + "=" + str(distance)
			min_distance = distance
			closest_string = tester

	return closest_string

def qwerty_find_closest_word(string1, dictionary):
	global insertionCost
	global deletionCost 
	global substitutionCost
	min_distance = 99999999
	closest_string = ""

	for tester in dictionary:
		distance = qwerty_levenshtein_distance(string1, tester, deletionCost,insertionCost)
		if (distance < min_distance):
			#print "found closer distance: " + string1 + "is similar to: " + tester
			#print "distance from " + string1 + " to " + tester + "=" + str(distance)
			min_distance = distance
			closest_string = tester

	return closest_string

def shiftInPlace(l, n):
	n = n % len(l)
	head = l[:n]
	l[:n] = []
	l.extend(head)
	return l

def qwerty_measure_error(typos, truewords, dictionarywords):
	
	error_count = 0
	tries = 0
	index = 0
	#error_individual = [0 for x in range(0, len(typos))]
	#rolling_window = [0 for x in range(0,20)]
	for typo in typos:
		solution = qwerty_find_closest_word(typo, dictionarywords) 
		#print "---------------------------------"
		#print "typo: " + typo + " correct would be: " + truewords[index] + " found: " + solution
		if (solution != truewords[index]):
			#print "This is WRONG"
			error_count += 1
		#else:	
			#print "This is CORRECT"
		
		index += 1
		tries += 1
		#print "STATISTICS:"
		#print "errors: " + str(error_count) + " out of: " + str(tries) + " remaining tests: " + str(len(typos)-tries)
		#error = (float(error_count)/float(tries))
		#error_individual[index-1] = error
		#rolling_window = shiftInPlace(rolling_window, 1)
		#rolling_window[-1] = error
		#print rolling_window

		#if ((max(rolling_window)-min(rolling_window) < 0.01) & (min(rolling_window) != 0)):
		#	print "found correct number of trials: " + str(tries)
		#	break
	
	#plt.plot([x for x in range(0,tries)], error_individual[0:tries])
	#plt.show()
	return float(error_count)/float(tries)


def measure_error(typos, truewords, dictionarywords):
	
	error_count = 0
	tries = 0
	index = 0
	#error_individual = [0 for x in range(0, len(typos))]
	#rolling_window = [0 for x in range(0,20)]
	for typo in typos:
		solution = find_closest_word(typo, dictionarywords) 
		#print "---------------------------------"
		#print "typo: " + typo + " correct would be: " + truewords[index] + " found: " + solution
		if (solution != truewords[index]):
			#print "This is WRONG"
			error_count += 1
		#else:	
			#print "This is CORRECT"
		
		index += 1
		tries += 1
		#print "STATISTICS:"
		#print "errors: " + str(error_count) + " out of: " + str(tries) + " remaining tests: " + str(len(typos)-tries)
		#error = (float(error_count)/float(tries))
		#error_individual[index-1] = error
		#rolling_window = shiftInPlace(rolling_window, 1)
		#rolling_window[-1] = error
		#print rolling_window

		#if ((max(rolling_window)-min(rolling_window) < 0.01) & (min(rolling_window) != 0)):
		#	print "found correct number of trials: " + str(tries)
		#	break
	
	#plt.plot([x for x in range(0,tries)], error_individual[0:tries])
	#plt.show()
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
	#"""
	##problem 1
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
	#"""
	############
	
		
	#time measuring
	#for i in range(0, len(typos)):
		#print str(typos[i]) + " corresponds to " + str(truewords[i])
	#print typos
	#print truewords
	
	#start = time.time()
	#print measure_error(typos, truewords, dictionary)
	#print "----------------------------"
	#print "----------------------------"
	#print "Time: " + str(time.time() -start)


	## setup the sample set of 193 typos and truewords
	"""
	#experiment prob 3

	complete = open("wikipediatypo.txt", "r")
	lines = complete.readlines()
	complete.close()
	typos = []
	truewords = []
	
	for line in lines:
		p = line.split()
		typos.append(str(p[0]))
		truewords.append(str(p[1]))


	idx = random.randint(0, len(typos)-81)
	typos = typos[idx:idx+81]
	truewords = truewords[idx:idx+81]
	print "starting parameter test"
	setting_iC = 0
	setting_dC = 0
	setting_sC = 0
	parameters = [0,1,2,4]
	min_error = 1
	start = time.time()
	for insertionCost in parameters:
		for deletionCost in parameters:
			for substitutionCost in parameters:
				error_local = measure_error(typos, truewords, dictionary)
				print "----------- NEW SETTING ----------------"
				if (error_local < min_error):
					min_error = error_local
					setting_iC = insertionCost
					setting_sC = substitutionCost
					setting_dC = deletionCost
					print "updated to setup: sC = " + str(setting_sC) + " dC = " + str(setting_dC) + " iC = " + str(setting_iC) + " with error rate = " + str(error_local)
				print "current setup: sC = " + str(substitutionCost) + " dC = " + str(deletionCost) + " iC = " + str(insertionCost) + " with error rate = " + str(error_local)

	print "-----------------FINAL-----------------"
	print "ideal setup: sC = " + str(setting_sC) + " dC = " + str(setting_dC) + " iC = " + str(setting_iC) + " with error rate = " + str(min_error)
	print "Time: " + str(time.time() -start)
	
	#print qwerty_distance('a','p')
	
	
	#experiment prob 4
	complete = open("wikipediatypo.txt", "r")
	lines = complete.readlines()
	complete.close()
	typos = []
	truewords = []
	
	for line in lines:
		p = line.split()
		typo_acc = ""
		for i in str(p[0]):
			if (i.isdigit() | i.isalpha()):
				typo_acc += i

			else:
				break 

		typos.append(str(typo_acc))
		truewords.append(str(p[1]))

	dictionary_clean = []
	for word in dictionary:
		word_acc = ""
		for i in word:
			if (i.isdigit() | i.isalpha()):
				word_acc += i

			else:
				break 
		dictionary_clean.append(str(word_acc))
	#print dictionary_clean
	#print typos
	idx = random.randint(0, len(typos)-54)
	typos = typos[idx:idx+54]
	truewords = truewords[idx:idx+54]

	print "starting parameter test"
	setting_iC = 0
	setting_dC = 0
	setting_sC = 0
	parameters = [1,2,4]
	min_error = 1
	start = time.time()
	for insertionCost in parameters:
		for deletionCost in parameters:
			error_local = qwerty_measure_error(typos, truewords, dictionary_clean)
			print "----------- NEW SETTING ----------------"
			if (error_local < min_error):
				min_error = error_local
				setting_iC = insertionCost
				setting_dC = deletionCost
				print "updated to setup: dC = " + str(setting_dC) + " iC = " + str(setting_iC) + " with error rate = " + str(error_local)
			print "current setup: dC = " + str(deletionCost) + " iC = " + str(insertionCost) + " with error rate = " + str(error_local)

	print "-----------------FINAL-----------------"
	print "ideal setup: dC = " + str(setting_dC) + " iC = " + str(setting_iC) + " with error rate = " + str(min_error)
	print "Time: " + str(time.time() -start)
	"""
	
	
