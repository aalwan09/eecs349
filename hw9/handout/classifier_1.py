import pickle
import sklearn
from sklearn import svm # this is an example of using SVM
from mnist import load_mnist
import matplotlib.pyplot as plt
import numpy as np
from experiments import fairtraintest


def preprocess(images):
    #this function is suggested to help build your classifier. 
    #You might want to do something with the images before 
    #handing them to the classifier. Right now it does nothing.
    
    ##we need to normalize the images
    
    length = len(images)
    processedimages = np.zeros((length, 28,28))


    for i in xrange(0, length):
        for j in xrange(0,28):
            for l in xrange(0,28):
                if images[i][j][l] > 0.5:
                     processedimages[i][j][l] = 1

    return processedimages

def build_classifier(images, labels, c, deg):
    #this will actually build the classifier. In general, it
    #will call something from sklearn to build it, and it must
    #return the output of sklearn. Right now it does nothing.
    classifier = svm.SVC(C=c, degree=deg)
    classifier.fit(images, labels)
    return classifier

##the functions below are required
def save_classifier(classifier, training_set, training_labels):
    #this saves the classifier to a file "classifier" that we will
    #load from. It also saves the data that the classifier was trained on.
    import pickle
    pickle.dump(classifier, open('classifier_1.p', 'w'))
    pickle.dump(training_set, open('training_set_1.p', 'w'))
    pickle.dump(training_labels, open('training_labels_1.p', 'w'))


def classify(images, classifier):
    #runs the classifier on a set of images. 
    return classifier.predict(images)

def error_measure(predicted, actual):
    return np.count_nonzero(abs(predicted - actual))/float(len(predicted))


def fivefoldsize(images, labels, C, degree):
        Imgslices = [images[j::5] for j in xrange(5)] #slices both our X and Y into n parts
        Labelslices = [labels[l::5] for l in xrange(5)]
        Error = np.zeros(5) #array of errors to collect after each fold

        for i in xrange(0, 5):
            Imgtraining = np.array(Imgslices[:i] + Imgslices[(i+1):]) #get the training sets by exluding one of the slices
            Labeltraining = np.array(Labelslices[:i] + Labelslices[(i+1):])
            Imgtraining.flatten() #formatting
            Labeltraining.flatten()
            print "fold: " + str(i)
            print "training classifier"

            ConcatenatedImages = np.concatenate((Imgtraining[0], Imgtraining[1], Imgtraining[2], Imgtraining[3]))
            ConcatenatedLabels = np.concatenate((Labeltraining[0], Labeltraining[1], Labeltraining[2], Labeltraining[3]))

            classifier = build_classifier(ConcatenatedImages, ConcatenatedLabels, C, degree)
            predicted = classify(Imgslices[i], classifier)
            error = error_measure(predicted, Labelslices[i])
            print "Error at fold " + str(i) +  " : " + str(error)
            Error[i] = error
        return np.average(Error)

def TrainingSizeFold(images, labels):
    TrainingSizes = [1000, 2000, 3000, 4000, 5000, 7500, 10000, 15000, 25000, 40000, 60000]
    SizeE = np.zeros(len(TrainingSizes))
    for i in xrange(0, len(TrainingSizes)): 
        print "Training on: " + str(TrainingSizes[i])
        newimages, newlabels, temp, temp2 = fairtraintest(images, labels, TrainingSizes[i], 0)
        SizeE[i] = fivefoldsize(newimages, newlabels, 1.0, 3)
        print "Avg Error at training size: " + str(SizeE[i])

    print SizeE[i]
    np.savetxt('SizeAnalysisSVM.txt', SizeE, delimiter=',') 

def TrainingCFold(images, labels):
    TrainingCvalues = [0.25, 0.5, 0.75, 1, 1.25, 2, 3]
    SizeE = np.zeros(len(TrainingCvalues))
    
    for i in xrange(0, len(TrainingCvalues)): 
        print "Training on: c = " + str(TrainingCvalues[i])
        SizeE[i] = fivefoldsize(SelectedImages, SelectedLabels, i, 3.0)
        print "Avg Error at training size: " + str(SizeE[i])

    print SizeE[i]
    np.savetxt('CAnalysisSVM.out', SizeE, delimiter=',') 


def TrainingDegreeFold(images, labels): 
    TrainingDvalues = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    SizeE = np.zeros(len(TrainingDvalues))
    
    for i in xrange(0, len(TrainingCvalues)): 
        print "Training on: degree = " + str(TrainingDvalues[i])
        SizeE[i] = fivefoldsize(SelectedImages, SelectedLabels, 1.0, i)
        print "Avg Error at training size: " + str(SizeE[i])

    print SizeE[i]
    np.savetxt('DegreeAnalysisSVM.out', SizeE, delimiter=',') 





if __name__ == "__main__":

    # Code for loading data
    images, labels = load_mnist(digits=range(0,10), path = '.')
    
    # preprocessing
    
    #aaa = preprocess(images)

    images = [i.flatten() for i in images]


    # print "label: " + str(labels[7413])
    # plt.imshow(np.reshape(images[7413], (28, 28)), cmap = 'binary', interpolation='nearest')
    # plt.show()
    
    # pick training and testing set
    # YOU HAVE TO CHANGE THIS TO PICK DIFFERENT SET OF DATA

    #TrainingSizeFold(images, labels)

    SelectedImages, SelectedLabels, temp, temp2 = fairtraintest(images, labels, 10000, 0)
    import pickle
    pickle.dump(SelectedImages, open('training_set_1.p', 'w'))
    pickle.dump(SelectedLabels, open('training_labels_1.p', 'w'))
    TrainingCFold(SelectedImages, SelectedLabels)
    TrainingDegreeFold(SelectedImages, SelectedLabels)

    # training_set = images[0:6000]
    # training_labels = labels[0:6000]
    # testing_set = images[6000:8000]
    # testing_labels = labels[6000:8000]

    # fairtraintest(images, labels, trainsize, testsize) - returns equally distributed Train and Test data 
    
    # Trainset, Trainlab, Testset, Testlab = fairtraintest(images, labels, 1587, 0)

    #TrainingSetPerformanceFair(images, labels)


    #build_classifier is a function that takes in training data and outputs an sklearn classifier.

    #classifier = build_classifier([images[:48000]], [labels[:48000]], 1.0, 3)
    # save_classifier(classifier, training_set, training_labels)
    # classifier = pickle.load(open('classifier_1.p'))
    #predicted = classify(images[48000:], classifier)
    #print error_measure(predicted, labels[48000:])
