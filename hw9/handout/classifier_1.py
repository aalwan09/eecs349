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

def preprocess(images):
    pass
    #this function is suggested to help build your classifier. 
    #You might want to do something with the images before 
    #handing them to the classifier. Right now it does nothing.
    ##we need to normalize the images

    #No need for preprocess on SVM as we use gradient 

def build_classifier(images, labels, c, deg, poly):
    #this will actually build the classifier. In general, it
    #will call something from sklearn to build it, and it must
    #return the output of sklearn. Right now it does nothing.

    if poly == 1:
        classifier = svm.SVC(C=c, degree=deg, kernel='poly')
        classifier.fit(images, labels)
    else:
        classifier = svm.SVC(C=c, degree=deg)
        classifier.fit(images, labels)
    return classifier

##the functions below are required
def save_classifier(classifier):
    #this saves the classifier to a file "classifier" that we will
    #load from. It also saves the data that the classifier was trained on.
    import pickle
    pickle.dump(classifier, open('classifier_1.p', 'w'))



def classify(images, classifier):
    #runs the classifier on a set of images. 
    return classifier.predict(images)

def error_measure(predicted, actual):
    return np.count_nonzero(abs(predicted - actual))/float(len(predicted))


def fivefoldsize(images, labels, C, degree, poly):
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

            classifier = build_classifier(ConcatenatedImages, ConcatenatedLabels, C, degree, poly)
            predicted = classify(Imgslices[i], classifier)
            error = error_measure(predicted, Labelslices[i])
            print "Error at fold " + str(i) +  " : " + str(error)
            Error[i] = error
        return np.average(Error)

def TrainingSizeFold(images, labels):
    TrainingSizes = [1000, 2000, 4000, 10000]
    SizeE = np.zeros(len(TrainingSizes))
    for i in xrange(0, len(TrainingSizes)): 
        print "Training on: " + str(TrainingSizes[i])
        newimages, newlabels, temp, temp2 = fairtraintest(images, labels, TrainingSizes[i], 0)
        #print len(newimages)
        #print len(newlabels)

        SizeE[i] = fivefoldsize(newimages, newlabels, 1.0, 3, -1)
        print "Avg Error at training size: " + str(SizeE[i])

    print SizeE[i]
    np.savetxt('SizeAnalysisSVM.txt', SizeE, delimiter=',') 

def TrainingCFold(images, labels):
    TrainingCvalues = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.25, 1.5, 1.75, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    SizeE = np.zeros(len(TrainingCvalues))
    
    for i in xrange(0, len(TrainingCvalues)): 
        print "Training on: c = " + str(TrainingCvalues[i])
        SizeE[i] = fivefoldsize(SelectedImages, SelectedLabels, TrainingCvalues[i], 3.0, -1)
        print "Avg Error at training size: " + str(SizeE[i])

    print SizeE[i]
    np.savetxt('CAnalysisSVM.txt', SizeE, delimiter=',') 

def TrainingPolyFold(images, labels): 
    TrainingPolyvalues = [0,1,2,3]
    SizeE = np.zeros(len(TrainingPolyvalues))
    
    for i in xrange(0, len(TrainingPolyvalues)): 
        print "Training on: degree = " + str(i)
        SizeE[i] = fivefoldsize(SelectedImages, SelectedLabels, 50, i, 1)
        print "Avg Error at training size: " + str(SizeE[i])

    print SizeE[i]
    np.savetxt('PolyAnalysisSVM.txt', SizeE, delimiter=',') 

def buildConfusionMatrix(predicted, actual):
    matrix = np.zeros((10,10))
    for i in xrange(0,len(predicted)):
        matrix[actual[i]][predicted[i]] += 1
    return matrix


def fivefoldconfusion(images, labels, C, degree, poly):

        Imgslices = [images[j::5] for j in xrange(5)] #slices both our X and Y into n parts
        Labelslices = [labels[l::5] for l in xrange(5)]
        Error = np.zeros(5) #array of errors to collect after each fold 
        myConfusionMatrixes = np.zeros((5,10,10))


        for i in xrange(0, 5):
            Imgtraining = np.array(Imgslices[:i] + Imgslices[(i+1):]) #get the training sets by exluding one of the slices
            Labeltraining = np.array(Labelslices[:i] + Labelslices[(i+1):])
            Imgtraining.flatten() #formatting
            Labeltraining.flatten()
            print "fold: " + str(i)
            print "training classifier"

            ConcatenatedImages = np.concatenate((Imgtraining[0], Imgtraining[1], Imgtraining[2], Imgtraining[3]))
            ConcatenatedLabels = np.concatenate((Labeltraining[0], Labeltraining[1], Labeltraining[2], Labeltraining[3]))

            classifier = build_classifier(ConcatenatedImages, ConcatenatedLabels, C, degree, poly)
            predicted = classify(Imgslices[i], classifier)

            error = error_measure(predicted, Labelslices[i])
            print "Error at fold " + str(i) +  " : " + str(error)
            myConfusionMatrixes[i] = buildConfusionMatrix(predicted, Labelslices[i])
            #print myConfusionMatrixes[i]


            for k in xrange(0, len(predicted)):
                if(predicted[k] != Labelslices[i][k]):
                    print "predicted: " + str(predicted[k])
                    print "actual: " + str(Labelslices[i][k])
                    plt.imshow(np.reshape(Imgslices[i][k], (28, 28)), cmap = 'binary', interpolation='nearest')
                    plt.show()

        


if __name__ == "__main__":

    # Code for loading data
    images, labels = load_mnist(digits=range(0,10), path = '.')
    
    #preprocessing
    #No preprocessing as SVM performs better with scaled weights

    images = [i.flatten() for i in images]

    #Training on different Sizes - ANALYSIS: 
    #TrainingSizeFold(images, labels)
    

    #picking training and testing set for optimizing SVM

    #SelectedImages, SelectedLabels, temp, temp2 = fairtraintest(images, labels, 1000, 0)

    import pickle
    #pickle.dump(SelectedImages, open('training_set_1.p', 'w'))
    #pickle.dump(SelectedLabels, open('training_labels_1.p', 'w'))

    SelectedImages = pickle.load(open('training_set_1_final.p'))
    SelectedLabels = pickle.load(open('training_labels_1_final.p'))


    #Training on C coefficient
    #TrainingCFold(SelectedImages, SelectedLabels)

    #Training on Poly Coefs
    #TrainingPolyFold(SelectedImages, SelectedLabels)
    
    #optimized classifier (from what we learnt in 5-fold): 
    #fivefoldconfusion(SelectedImages, SelectedLabels, 50, 2, 1)

    #Error = fivefoldsize(SelectedImages, SelectedLabels, , 3)
    #classifier = build_classifier(SelectedImages, SelectedLabels, 50, 2, 1)
    #save_classifier(classifier)
    
