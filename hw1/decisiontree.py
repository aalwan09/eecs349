import csv
import sys
import random
import math

def calc_E(freq_true, freq_false):
		if ((freq_true == 0) | (freq_false == 0)):
			return 0
		else:
			try:
				return -1*(float(freq_true)*math.log(float(freq_true), 2)) - (float(freq_false)*math.log(float(freq_false), 2))
			except:
				print "freq_true: " + str(freq_true)
				print "freq_false: " + str(freq_false)

class Attribute:
	informationGain = 0
	entropyToTarget = 0
	name = ""
	index_in_data = 0
	def __init__(self, name, tt, tf, ft, ff, entropyOfTarget, index):
		#print "In Attribute: "
		total = tt+tf+ft+ff
		self.name = name
		"""
		print "total: "
		print total
		print "tt: "
		print tt
		print "tf: "
		print tf
		print "ft: "
		print ft
		print "ff: "
		print ff
		"""
		self.index_in_data = index
		
		if ((tt+tf) == 0):
			P_tt = 0.
			P_tf = 0.
		else: 
			P_tt = float(tt)/float(tt+tf)	
			P_tf = float(tf)/float(tt+tf)

		if ((ft+ff) == 0):
			P_ft = 0.
			P_ff = 0.
		else: 
			P_ft = float(ft)/float(ft+ff)	
			P_ff = float(ff)/float(ft+ff)

				
		
		self.entropyToTarget = float(tt+tf)/float(total) * calc_E(P_tt, P_tf) + float(ft+ff)/float(total) * calc_E(P_ft, P_ff)
		self.informationGain = entropyOfTarget - self.entropyToTarget
	def print_me(self):
		print "Attribute Name: " + self.name
		print "\t Information Gain: " + str(self.informationGain)
		print "\t Entropy w.r.t target: " + str(self.entropyToTarget)
		print "\t Index in data: " + str(self.index_in_data)
		
		

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
		self.runTree()

	def runTree(self):
		f = open(self.inputFileName, 'rb');
		self.reader = csv.reader(f, delimiter='\t')
		if (self.process_text()):
			self.calc_prior()
			self.calc_E_T()	
			index = 0
			for x in self.categories:
				tt = 0 
				tf = 0
				ft = 0
				ff = 0
				for y in self.trainingSet:
					if (y[index] == "true "):
						if (y[len(self.categories)] == "true"):
							tt += 1
						elif (y[len(self.categories)] == "false"):
							tf += 1
					elif (y[index] == "false "):
						if (y[len(self.categories)] == "true"):
							ft += 1
						elif (y[len(self.categories)] == "false"):
							ff += 1
					else:
						sys.exit("Error 2, unidentified classifier")
				self.attributes.append(Attribute(x, tt, tf, ft, ff, self.entropyOfTarget,index))
				index += 1
				
			for element in self.attributes:
				element.print_me()
		f.close() 
		del self.trainingSet[:]
		del self.testingSet[:]
		del self.categories[:]
		del self.attributes[:]
		
				
				
	
	def print_input(self):
		for row in self.reader:
			print row
	
	def process_text(self):
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
			return 1
		except: 
			sys.exit("Error processing text file")
			return 0

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
		
	def calc_E_T(self):
		freq_true = float(self.numberTrue)/float(self.trainingSetSize)
		freq_false = float(self.numberFalse)/float(self.trainingSetSize)
		self.entropyOfTarget = calc_E(freq_true, freq_false)
		print "Entropy of Target Concept: " + str(self.entropyOfTarget)


	



if __name__ == "__main__":
	if ((len(sys.argv) < 5) | (len(sys.argv) > 5)):
		sys.exit( "Wrong number of command line arguments. \n Usage: python decisiontree.py <inputFileName> <trainingSetSize> <numberOfTrials> <verbose> ")
	else:
		try:
			inputFileName = str(sys.argv[1])
			trainingSetSize = int(sys.argv[2])
			numberOfTrials = int(sys.argv[3])
			verbose = bool(sys.argv[4])
		except:
			sys.exit("The provided arguments are malformed")
	for x in range(1,numberOfTrials+1):
		print "TRIAL: " + str(x)
    		solver = ID3Solver(inputFileName, trainingSetSize, numberOfTrials, verbose)
		del solver
