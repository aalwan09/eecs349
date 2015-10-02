import csv
import sys
import random
import math

class node:
	parent = ""
	trueChild = ""
	falseChihld = ""



class ID3Solver:
	inputFileName = ""
	trainingSetSize = 0
	numberOfTrials = 0
	verbose = False
	dataSize = 0

	trainingSet = []
	testingSet = []
	categories = []

	reader = None

	numberTrue = 0
	numberFalse = 0
	
	expectedPPTrue = 0;
	expectedPPFalse = 0;

	entropyOfTarget = 0;

	attributes = []

	
	def __init__(self, inputFileName, trainingSetSize, numberOfTrials, verbose):
		self.inputFileName = inputFileName
		self.trainingSetSize = trainingSetSize
		self.numberOfTrials = numberOfTrials
		self.verbose = verbose
		
		print "Starting ID3 Solver with following arguments:"
		print "Input file: " + self.inputFileName
		print "Trainingset size: " + str(self.trainingSetSize)
		print "Number of trials: " + str(self.numberOfTrials)
		if (self.verbose):
			print "RUNNING VERBOSE MODE"
		f = open(self.inputFileName, 'rb');
		self.reader = csv.reader(f, delimiter='\t')
		try:
			self.categories = next(self.reader)
			del self.categories[-1]
			entire_data_list = list(self.reader)	
			self.dataSize = len(entire_data_list)
			for x in range(0,self.trainingSetSize):
				rand_i = random.randrange(0,len(entire_data_list))
				self.trainingSet.append(entire_data_list[rand_i])
				del entire_data_list[rand_i]
			self.testingSet = entire_data_list
			print "size of testing set: " + str(len(self.testingSet))
			print "size of training set: " + str(len(self.trainingSet))
			print "size of all data: " + str(self.dataSize)
		except: 
			sys.exit("Error processing text file")

		print self.categories
		print self.trainingSet
		
		self.calc_prior()

		
		###call the recursive function in here



		f.close() 
		del self.trainingSet[:]
		del self.testingSet[:]
		del self.categories[:]
		del self.attributes[:]

	def calc_prior(self):
		#print self.trainingSet
		for x in self.trainingSet:
			if (x[len(self.categories)] == "true"):
				self.numberTrue += 1
			elif (x[len(self.categories)] == "false"):
				self.numberFalse += 1
			else:
				sys.exit("Error, unidentified classifier")
		#print "number of trues: " + str(self.numberTrue)
		#print "number of falses: " + str(self.numberFalse)
		self.expectedPPTrue = float(self.numberTrue)/float(self.trainingSetSize)
		self.expectedPPFalse = float(self.numberFalse)/float(self.trainingSetSize)
		if ((self.expectedPPTrue+self.expectedPPFalse) != 1.):
			print "something went wrong. PP true: " + str( self.expectedPPTrue) + " PP wrong: " + str(self.expectedPPFalse)
		print "Prior Probability for TRUE: " + str(self.expectedPPTrue*100.) + "%"
		print "Prior Probability for FALSE: " + str(self.expectedPPFalse*100.) + "%"

		






if __name__ == "__main__":
	if ((len(sys.argv) < 5) | (len(sys.argv) > 5)):
		sys.exit( "Wrong number of command line arguments. \n Usage: python decisiontree.py <inputFileName> <trainingSetSize> <numberOfTrials> <verbose> ")
	else:
		try:
			inputFileName = str(sys.argv[1])
			trainingSetSize = int(sys.argv[2])
			numberOfTrials = int(sys.argv[3])
			if (sys.argv[4] == "0"):
				verbose = False
			else:
				verbose = True
		except:
			sys.exit("The provided arguments are malformed")
	for x in range(1,numberOfTrials+1):
		print "TRIAL: " + str(x)
    		solver = ID3Solver(inputFileName, trainingSetSize, numberOfTrials, verbose)
		del solver
