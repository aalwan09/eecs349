#Starter code for spam filter assignment in EECS349 Machine Learning
#Author: Prem Seetharaman (replace your name here)

import sys
import numpy as np
import os
import shutil
import math as m

def parse(text_file):
	#This function parses the text_file passed into it into a set of words. Right now it just splits up the file by blank spaces, and returns the set of unique strings used in the file. 
	content = text_file.read()
	return np.unique(content.split())

def writedictionary(dictionary, dictionary_filename):
	#Don't edit this function. It writes the dictionary to an output file.
	output = open(dictionary_filename, 'w')
	header = 'word\tP[word|spam]\tP[word|ham]\n'
	output.write(header)
	for k in dictionary:
		line = '{0}\t{1}\t{2}\n'.format(k, str(dictionary[k]['spam']), str(dictionary[k]['ham']))
		output.write(line)
		

def makedictionary(spam_directory, ham_directory, dictionary_filename):
	#Making the dictionary. 
	spam = [f for f in os.listdir(spam_directory) if os.path.isfile(os.path.join(spam_directory, f))]
	ham = [f for f in os.listdir(ham_directory) if os.path.isfile(os.path.join(ham_directory, f))]
	
	spam_prior_probability = len(spam)/float((len(spam) + len(ham)))
	
	
	words = {}

	#These for loops walk through the files and construct the dictionary. The dictionary, words, is constructed so that words[word]['spam'] gives the probability of observing that word, given we have a spam document P(word|spam), and words[word]['ham'] gives the probability of observing that word, given a hamd document P(word|ham). Right now, all it does is initialize both probabilities to 0. TODO: add code that puts in your estimates for P(word|spam) and P(word|ham).
	
	for s in spam:
		previous_words = []
		for word in parse(open(spam_directory + s)):
			if word not in words:
				words[word] = {'spam': 1, 'ham': 0}
			elif word not in previous_words:
				words[word]['spam'] += 1
				previous_words.append(word)

	for h in ham:
		previous_words = []
		for word in parse(open(ham_directory + h)):
			if word not in words:
				words[word] = {'spam': 0, 'ham': 1}
			elif word not in previous_words:
				words[word]['ham'] += 1
				previous_words.append(word)

	for word in words:
		if (words[word]['spam'] == 0):
			words[word]['spam'] = float(1)/float(len(spam)+1)
		else:
			words[word]['spam'] = float(words[word]['spam'])/float(len(spam))
		if (words[word]['ham'] == 0):
			words[word]['ham'] = float(1)/float(len(ham)+1)
		else:
			words[word]['ham'] = float(words[word]['ham'])/float(len(ham))
		
		
	
	#Write it to a dictionary output file.
	writedictionary(words, dictionary_filename)
	
	return words, spam_prior_probability

def is_spam(content, dictionary, spam_prior_probability):
	#TODO: Update this function. Right now, all it does is checks whether the spam_prior_probability is more than half the data. If it is, it says spam for everything. Else, it says ham for everything. You need to update it to make it use the dictionary and the content of the mail. Here is where your naive Bayes classifier goes.
	prob_spam = m.log(spam_prior_probability,10)
	prob_ham = m.log(spam_prior_probability,10)
	
	add_spam = 0
	add_ham = 0
	for word in content:
		if word in dictionary:
			add_spam += m.log(dictionary[word]['spam'],10)
			add_ham += m.log(dictionary[word]['ham'],10)
		else:
			add_spam += m.log(spam_prior_probability,10)
			add_spam += m.log((1-spam_prior_probability),10)

	prob_spam += add_spam
	prob_ham += add_ham

	if prob_spam > prob_ham:
		return 1
	else:
		return 0
	
	



def spamsort(mail_directory, spam_directory, ham_directory, dictionary, spam_prior_probability):
	spam_count = 0
	mail = [f for f in os.listdir(mail_directory) if os.path.isfile(os.path.join(mail_directory, f))]
	for m in mail:
		content = parse(open(mail_directory + m))
		spam = is_spam(content, dictionary, spam_prior_probability)
		if spam:
			shutil.copy(mail_directory + m, spam_directory)
			spam_count += 1
		else:
			shutil.copy(mail_directory + m, ham_directory)
	return spam_count

def valid_path(dir_path, filename):
	full_path = os.path.join(dir_path, filename)
	return os.path.isfile(full_path)




if __name__ == "__main__":
	#Here you can test your functions. Pass it a training_spam_directory, a training_ham_directory, and a mail_directory that is filled with unsorted mail on the command line. It will create two directories in the directory where this file exists: sorted_spam, and sorted_ham. The files will show up  in this directories according to the algorithm you developed.
	training_spam_directory = sys.argv[1]
	training_ham_directory = sys.argv[2]
	
	test_mail_directory = sys.argv[3]
	test_spam_directory = 'sorted_spam'
	test_ham_directory = 'sorted_ham'

	exp_spam_training = "exp_spam_training/"
	exp_spam_testing = "exp_spam_testing/"
	exp_ham_training = "exp_ham_training/"
	exp_ham_testing = "exp_ham_testing/"
	
	directory = [exp_spam_training, exp_spam_testing, exp_ham_training, exp_ham_testing]
	
	if not os.path.exists(test_spam_directory):
		os.mkdir(test_spam_directory)
	if not os.path.exists(test_ham_directory):
		os.mkdir(test_ham_directory)
	
	dictionary_filename = "dictionary.dict"

	for d in directory:
		if os.path.exists(d):
			shutil.rmtree(d)
		os.mkdir(d)

	src_files = [ f for f in os.listdir(training_spam_directory) if os.path.isfile(os.path.join(training_spam_directory,f))]
	spam_files = [f for f in src_files if valid_path(training_spam_directory, f)]
	src_files = [ f for f in os.listdir(training_ham_directory) if os.path.isfile(os.path.join(training_ham_directory,f))]
	ham_files = [f for f in src_files if valid_path(training_ham_directory, f)]

	N = 10

	prob_spam = len(spam_files)/float(len(ham_files)+len(spam_files))
	prob_ham = len(ham_files)/float(len(ham_files)+len(spam_files))
	group_size = (len(spam_files)+len(ham_files))/N

	print "using " + str(N) + " fold cross validation with one group containing " + str(group_size) + " items"

	number_files_spam = int(group_size*prob_spam)
	number_files_ham = int(group_size*prob_ham)
	#print "using " +str(number_files_spam) + " samples for spam and " + str(number_files_ham) + " for ham per group"
	switch_bool = 0

	while ((number_files_spam + number_files_ham) < group_size):	
		if switch_bool:
			number_files_spam += 1
			switch_bool = 0
		else:
			number_files_ham += 1
			switch_bool = 1

	print "using " +str(number_files_spam) + " samples for spam and " + str(number_files_ham) + " for ham "



	for n in range(0,N):
		
		testing_files_spam = spam_files[n*number_files_spam:(n+1)*number_files_spam]
		testing_files_ham = ham_files[n*number_files_ham:(n+1)*number_files_ham]
		training_files_spam = spam_files[0:n*number_files_spam] + spam_files[(n+1)*number_files_spam:N*number_files_spam]
		training_files_ham = ham_files[0:n*number_files_ham] + ham_files[(n+1)*number_files_ham:N*number_files_ham]
		print "------"
		print len(testing_files_spam)
		print len(testing_files_ham)
		print len(testing_files_spam) + len(testing_files_ham)
		print len(training_files_spam) + len(training_files_ham)

	
	#create the dictionary to be used
	dictionary, spam_prior_probability = makedictionary(training_spam_directory, training_ham_directory, dictionary_filename)
	#sort the mail
	spamsort(test_mail_directory, test_spam_directory, test_ham_directory, dictionary, spam_prior_probability) 
