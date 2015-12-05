import pickle
import sklearn
from sklearn import svm # this is an example of using SVM
from mnist import load_mnist


def preprocess(images):
    #this function is suggested to help build your classifier. 
    #You might want to do something with the images before 
    #handing them to the classifier. Right now it does nothing.
    return [i.flatten() for i in images]

def build_classifier(images, labels):
    #this will actually build the classifier. In general, it
    #will call something from sklearn to build it, and it must
    #return the output of sklearn. Right now it does nothing.
    classifier = svm.SVC()
    classifier.fit(images, labels)
    return classifier

##the functions below are required
def save_classifier(classifier, training_set, training_labels):
    #this saves the classifier to a file "classifier" that we will
    #load from. It also saves the data that the classifier was trained on.
    import pickle
    pickle.dump(classifier, open('classifier_2.p', 'w'))
    pickle.dump(training_set, open('training_set.p', 'w'))
    pickle.dump(training_labels, open('training_labels.p', 'w'))


def classify(images, classifier):
    #runs the classifier on a set of images. 
    return classifier.predict(images)

def error_measure(predicted, actual):
    return np.count_nonzero(abs(predicted - actual))/float(len(predicted))

def buildConfusionMatrix(predicted, actual):
	matrix = np.zeros((10,10))
	for i in xrange(0,len(predicted)):
		matrix[actual[i]][predicted[i]] += 1
	return matrix

if __name__ == "__main__":

    # Code for loading data
    
    # preprocessing
    images = preprocess(images)
    
    # pick training and testing set
    # YOU HAVE TO CHANGE THIS TO PICK DIFFERENT SET OF DATA
    training_set = images[0:1000]
    training_labels = labels[0:1000]
    testing_set = images[-100:]
    testing_labels = labels[-100:]

    #build_classifier is a function that takes in training data and outputs an sklearn classifier.
    classifier = build_classifier(training_set, training_labels)
    save_classifier(classifier, training_set, training_labels)
    classifier = pickle.load(open('classifier'))
    predicted = classify(testing_set, classifier)
    print error_measure(predicted, testing_labels)

    # print "label: " + str(labels[7413])
    # plt.imshow(np.reshape(images[7413], (28, 28)), cmap = 'binary', interpolation='nearest')
    # plt.show()
