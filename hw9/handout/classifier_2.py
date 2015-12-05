import pickle
import sklearn
from sklearn import svm # this is an example of using SVM
from sklearn.neighbors import KNeighborsClassifier
from mnist import load_mnist
import numpy as np
import matplotlib.pyplot as plt

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




def preprocess(images):
    #this function is suggested to help build your classifier. 
    #You might want to do something with the images before 
    #handing them to the classifier. Right now it does nothing.
    
    ##we need to normalize the images
    





    length = len(images)
    processedimages = np.zeros((length, 784))
    #print length
    #print images[0][0][0]
    #print images[0]
    #print len(images[0])
    print length
    for i,im in enumerate(images):
    	print len(images)
    	print len(images[0])
    	print len(images[0][0])
        for j in range(0,28):
        	for k in range(0,28):
        		if images[i][j][k] > 0.5:
        			processedimages[i][j*28+k] = 1.
        		
	return processedimages

def build_classifier(images, labels, K):
    #this will actually build the classifier. In general, it
    #will call something from sklearn to build it, and it must
    #return the output of sklearn. Right now it does nothing.
    classifier = KNeighborsClassifier(n_neighbors = K)
    classifier.fit(images, labels)
    return classifier

##the functions below are required
def save_classifier(classifier, training_set, training_labels):
    #this saves the classifier to a file "classifier" that we will
    #load from. It also saves the data that the classifier was trained on.
    import pickle
    pickle.dump(classifier, open('classifier_2.p', 'w'))
    pickle.dump(training_set, open('training_set_2.p', 'w'))
    pickle.dump(training_labels, open('training_labels_2.p', 'w'))


def classify(images, classifier):
    #runs the classifier on a set of images. 
    return classifier.predict(images)

def error_measure(predicted, actual):
    return np.count_nonzero(abs(predicted - actual))/float(len(predicted))


def fivefoldsize(images, labels, K):
        Imgslices = [images[j::5] for j in xrange(5)] #slices both our X and Y into n parts
        Labelslices = [labels[l::5] for l in xrange(5)]
        Score = np.zeros(5) #array of errors to collect after each fold 

        for i in xrange(0, 5):
			Imgtraining = np.array(Imgslices[:i] + Imgslices[(i+1):]) #get the training sets by exluding one of the slices
			Labeltraining = np.array(Labelslices[:i] + Labelslices[(i+1):])
			Imgtraining.flatten() #formatting
			Labeltraining.flatten()
			print "fold: " + str(i)
			print "training classifier"

			ConcatenatedImages = np.concatenate((Imgtraining[0], Imgtraining[1], Imgtraining[2], Imgtraining[3]))
			ConcatenatedLabels = np.concatenate((Labeltraining[0], Labeltraining[1], Labeltraining[2], Labeltraining[3]))

			classifier = build_classifier(ConcatenatedImages, ConcatenatedLabels, K)
			#print Imgslices[i][0]
			#print Labelslices[i][0]
			#plt.imshow(Imgslices[i][0], cmap = 'binary', interpolation="nearest")
			#plt.show()
			predictions = classifier.predict(Imgslices[i])
			error_count = 0
			print Labelslices[i]
			print predictions
			for j,p in enumerate(predictions):
				if (p != Labelslices[i][j]):
					error_count += 1
			print "Error at fold " + str(i) +  " : " + str(float(error_count)/len(predictions))
			Score[i] = float(error_count/len(predictions))
        return np.average(Score)

def TrainingSizeFold(images, labels):
    TrainingSizes = [1000, 2000, 3000, 4000, 5000, 7500, 10000, 15000, 25000, 40000, 60000]
    SizeS = np.zeros(len(TrainingSizes))
    for i in xrange(0, len(TrainingSizes)): 
        print "Training on: " + str(TrainingSizes[i])
        newimages, newlabels, temp, temp2 = fairtraintest(images, labels, TrainingSizes[i], 0)
        SizeS[i] = fivefoldsize(newimages, newlabels, 3)
        print "Avg Score at training size: " + str(SizeS[i])

    print SizeS[i]
    np.savetxt('SizeAnalysisKNN.txt', SizeS, delimiter=',') 

def TrainingKFold(images, labels):
    # TrainingCvalues = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.25, 1.5, 1.75, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    TrainingKvalues = [1, 3, 5, 10]
    print TrainingCvalues
    SizeS = np.zeros(len(TrainingKvalues))
    
    for i in xrange(0, len(TrainingKvalues)): 
        print "Training on: K = " + str(TrainingKvalues[i])
        SizeS[i] = fivefoldsize(SelectedImages, SelectedLabels, TrainingKvalues[i])
        print "Avg Error at training size: " + str(SizeS[i])

    print SizeS[i]
    np.savetxt('CAnalysisKNN.out', SizeS, delimiter=',') 


def buildConfusionMatrix(predicted, actual):
    matrix = np.zeros((10,10))
    for i in xrange(0,len(predicted)):
        matrix[actual[i]][predicted[i]] += 1
    return matrix


if __name__ == "__main__":

    # Code for loading data
    images, labels = load_mnist(digits=range(0,10), path = '.')

    #preprocessing
    #need to bring images into vector format
    images = [i.flatten() for i in images]
    #print images
    

 
    #Training on different Sizes - ANALYSIS: 
    TrainingSizeFold(images, labels)
    

    # picking training and testing set for optimizing SVM

    #SelectedImages, SelectedLabels, temp, temp2 = fairtraintest(images, labels, 10000, 0)
    #import pickle
    #pickle.dump(SelectedImages, open('training_set_2.p', 'w'))
    #pickle.dump(SelectedLabels, open('training_labels_2.p', 'w'))

    #SelectedImages = pickle.load(open('training_set_1_final.p'))
    #SelectedLabels = pickle.load(open('training_labels_1_final.p'))

    #Training on C coefficient
    #TrainingCFold(SelectedImages, SelectedLabels)
    
    #optimized classifier (from what we learnt in 5-fold): 
    #Error = fivefoldsize(SelectedImages, SelectedLabels,bestc, 3)
    #save_classifier(classifier, training_set, training_labels)
    
