import csv
import sys
import random
import math

tree = []

def calc_E(freq_true, freq_false):
    if ((freq_true == 0) | (freq_false == 0)):
        return 0
    else:
        try:
            return -1*(float(freq_true)*math.log(float(freq_true), 2)) - (float(freq_false)*math.log(float(freq_false), 2))
        except:
            print "freq_true: " + str(freq_true)
            print "freq_false: " + str(freq_false)
            
def findHighestIG(local_trainingSet, local_categories, entropyOfTarget):
	index = 0
	attributes = []
	max_information_gain = 0.;
	max_information_gain_index = 0;
	#print "trying to find highest IG"
	tt = 0
	tf = 0
	ft = 0
	ff = 0
	for x in local_categories:
		tt = 0
		tf = 0
		ft = 0
		ff = 0
		for y in local_trainingSet:
			if (y[index] == "true "):
				if (y[-1] == "true"):
					tt += 1
				elif (y[-1] == "false"):
					tf += 1
			elif (y[index] == "false "):
				if (y[-1] == "true"):
					ft += 1
				elif (y[-1] == "false"):
					ff += 1
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
			
		total = tt+tf+ft+ff
		entropyToTarget = float(tt+tf)/float(total) * calc_E(P_tt, P_tf) + float(ft+ff)/float(total) * calc_E(P_ft, P_ff)
		informationGain = entropyOfTarget - entropyToTarget
		if (informationGain > max_information_gain):
			max_information_gain = informationGain
			max_information_gain_index = index
		
		#print str(local_categories[index]) + " Information Gain = " + str(informationGain)
		index += 1
	#print "MAX IG NODE: " + str(local_categories[max_information_gain_index])
	highestIGnode = node("null", local_categories[max_information_gain_index], "null", "null")
	highestIGnode.informationGain = max_information_gain
	highestIGnode.tt = tt
	highestIGnode.tf = tf
	highestIGnode.ft = ft
	highestIGnode.ff = ff
	return highestIGnode
	
def	trueTraining(input_trainingSet, input_categories, attribute):
	index = 0
	#print "finding trueTrainign"
	#print input_categories
	for x in input_categories:
		#print x
		if (x == str(attribute)):
			#print "found attribute in categories, index: " + str(index)
			break
		else:
			index += 1
	output_trainingSet = []
	for x in input_trainingSet:
		if (x[index] == "true "):
			output_trainingSet.append(x[0:index] + x[index+1:len(x)])	
	return output_trainingSet

def	falseTraining(input_trainingSet, input_categories, attribute):
	index = 0
	for x in input_categories:
		if (x == str(attribute)):
			break
		else:
			index += 1
	output_trainingSet = []
	for x in input_trainingSet:
		if (x[index] == "false "):
			output_trainingSet.append(x[0:index] + x[index+1:len(x)])
	return output_trainingSet
	
def update_categories(input_categories, attribute):
	index = 0
	#print "input to categories update: "
	#print input_categories
	#print "looking for: " + str(attribute)
	for x in input_categories:
		if (x == str(attribute)):
			#print "found attribute"
			break
		else:
			index += 1
	output = (input_categories[0:index] + input_categories[index+1:len(input_categories)])
	#print "output from categories update: "
	#print output
	return output
	
	
def update_E_T(trainingSet, categories):
	numberTrue = 0
	numberFalse = 0
	for x in trainingSet:
		if (x[-1] == "true"):
			numberTrue += 1
		elif (x[-1] == "false"):
			numberFalse += 1
		else:
			sys.exit("Error, unidentified classifier")
	freq_true = float(numberTrue)/float(len(trainingSet))
	freq_false = float(numberFalse)/float(len(trainingSet))
	out = calc_E(freq_true, freq_false)
	print "Entropy of Target: " + str(out)
	return out

class node:
    parent = ""
    attribute = ""
    trueChild = ""
    falseChild = ""
    informationGain = 0.
    tt = 0
    tf = 0
    ft = 0
    ff = 0
    def __init__(self, parent, attribute, trueChild, falseChild):
        self.parent = parent
        self.attribute = attribute
        self.trueChild = trueChild
        self.falseChild = falseChild
    def print_name():
        print self.attribute
        


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
		global tree
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
		if (verbose):
			print self.categories
			print self.trainingSet
		
		self.calc_prior()
		
		self.calc_E_T();
        
		#testNode = self.findHighestIG(self.trainingSet, self.categories, self.entropyOfTarget)
		#print testNode.attribute
		
		###call the recursive function in here
		rootNode = self.createTree(self.trainingSet, self.categories, "root")
		
		


		f.close()
		del self.trainingSet[:]
		del self.testingSet[:]
		del self.categories[:]
		del self.attributes[:]
		
	def createTree(self, local_trainingSet, local_categories, parent):
		self.entropyOfTarget = update_E_T(local_trainingSet, local_categories)
		newNode = findHighestIG(local_trainingSet, local_categories, self.entropyOfTarget)
		newNode.parent = parent
		print "New Node: Parent: " + str(parent) + " Attribute: " +str(newNode.attribute)
		#print "Passed categories: " + str(local_categories)
		trueChild_trainingSet = trueTraining(local_trainingSet, local_categories, newNode.attribute)
		falseChild_trainingSet = falseTraining(local_trainingSet, local_categories, newNode.attribute)
		new_categories = update_categories(local_categories, newNode.attribute)
		if (len(new_categories) == 1):
			finalNode = node(str(newNode.attribute), str(new_categories[0]), "", "")
			return finalNode
		elif ((self.entropyOfTarget-newNode.informationGain)!=0.):
			newNode.trueChild = self.createTree(trueChild_trainingSet, new_categories, newNode.attribute)
			newNode.falseChild = self.createTree(falseChild_trainingSet, new_categories, newNode.attribute)
		elif ((self.entropyOfTarget-newNode.informationGain) == 0.0):
			if (newNode.tt > newNode.tf):
				newNode.trueChild = "True"
			else:
				newNode.trueChild = "False"
			if (newNode.ft > newNode.ff):
				newNode.falseChild = "True"
			else:
				newNode.falseChild = "False"
		return newNode 

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
