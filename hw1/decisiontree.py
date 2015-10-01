import csv
import sys
import random
import math



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

	
	def __init__(self):

		if ((len(sys.argv) < 5) | (len(sys.argv) > 5)):
			sys.exit( "Wrong number of command line arguments. \n Usage: python decisiontree.py <inputFileName> <trainingSetSize> <numberOfTrials> <verbose> ")
		else:
			try:
				self.inputFileName = str(sys.argv[1])
				self.trainingSetSize = int(sys.argv[2])
				self.numberOfTrials = int(sys.argv[3])
				self.verbose = bool(sys.argv[4])
			except:
				sys.exit("The provided arguments are malformed")
		self.reader = csv.reader(open(self.inputFileName, 'rb'), delimiter='\t')
		self.runTree()

	def runTree(self):
		if (self.process_text()):
			self.calc_prior()
			self.calc_E_T()	
	
	def print_input(self):
		for row in self.reader:
			print row
	
	def process_text(self):
		try:
			self.categories = next(self.reader)
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
			return 1
		except: 
			sys.exit("Error processing text file")
			return 0

	def calc_prior(self):
		print self.trainingSet
		for x in self.trainingSet:
			if (x[len(self.categories)-1] == "true"):
				self.numberTrue += 1
			elif (x[len(self.categories)-1] == "false"):
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
		
	def calc_E_T(self):
		freq_true = float(self.numberTrue)/float(self.trainingSetSize)
		freq_false = float(self.numberFalse)/float(self.trainingSetSize)
		self.entropyOfTarget = -1*(freq_true*math.log(freq_true, 2)) - (freq_false*math.log(freq_false, 2))
		print self.entropyOfTarget
	



if __name__ == "__main__":
    solver = ID3Solver()
