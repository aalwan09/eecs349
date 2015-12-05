import pickle
import sklearn
from sklearn import svm # this is an example of using SVM
from mnist import load_mnist
import matplotlib.pyplot as plt
import numpy as np

#fairtraintest return Training and Testing Arrays of size trainsize and testsize (x28x28)
#it selects example so as to have an equal distribution of each label - You can also put 0 for testsize or trainsize, and then it just returns an equal distribution of labels
def fairtraintest(images, labels, trainsize, testsize): 
	if len(images) < trainsize + testsize:
		print "error: cannot divide images without repeats"
		return False
	else: 
		indexes = np.arange(len(images))
		np.random.shuffle(indexes) #randomize data selected
		
		done = False

		doneTest = np.zeros(10, dtype=bool)
		Testcounts = np.zeros(10)
		Testtotalcount = 0
		
		TestingSet = np.zeros((testsize, 784))
		TestingLabels = np.zeros(testsize)

		if(testsize == 0):
			doneTest = np.ones(10, dtype=bool)

		doneTraining = np.zeros(10, dtype=bool)
		Traincounts = np.zeros(10)
		Traintotalcount = 0

		TrainingSet = np.zeros((trainsize, 784))
		TrainLabels = np.zeros(trainsize)

		if(trainsize == 0):
			doneTraining = np.ones(10, dtype=bool)

		count = 0
		while done == False:
			label = labels[indexes[count]]
			if doneTest[label] == False and Testtotalcount < testsize and testsize > 0:
				TestingSet[Testtotalcount] = images[indexes[count]]
				TestingLabels[Testtotalcount] = label
				Testtotalcount += 1
				Testcounts[label] += 1

				if Testcounts[label] >= (testsize/10):
					doneTest[label] = True

			elif doneTraining[label] == False and Traintotalcount < trainsize and trainsize > 0:
				TrainingSet[Traintotalcount] = images[indexes[count]]
				TrainLabels[Traintotalcount] = label
				Traintotalcount += 1
				Traincounts[label] += 1 

				if Traincounts[label] >= (trainsize/10):
					doneTraining[label] = True

			if np.all(doneTest) and np.all(doneTraining):
				done = True

			count += 1 

			if count == len(images):
				done = True

		return TrainingSet, TrainLabels, TestingSet, TestingLabels


