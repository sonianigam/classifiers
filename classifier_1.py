import pickle
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from mnist import load_mnist
import numpy as np


def preprocess(images):
    #this function is suggested to help build your classifier. 
    #You might want to do something with the images before 
    #handing them to the classifier. Right now it does nothing.
    return np.array([i.flatten() for i in images])

def build_classifier(images, labels):
    #this will actually build the classifier. In general, it
    #will call something from sklearn to build it, and it must
    #return the output of sklearn. Right now it does nothing.
    classifier = KNeighborsClassifier(n_neighbors=10)

    #labels = np.array(labels)
    #images = np.array(images)
    labels.shape = labels.shape[0]

    classifier.fit(images, labels)

    return classifier

##the functions below are required
def save_classifier(classifier, training_set, training_labels):
    #this saves the classifier to a file "classifier" that we will
    #load from. It also saves the data that the classifier was trained on.
    import pickle
    pickle.dump(classifier, open('classifier_1.p', 'wb'))



def classify(images, classifier):
    #runs the classifier on a set of images. 
    "entered prediction"
    return classifier.predict(images)

def error_measure(predicted, actual):
    return np.count_nonzero(abs(predicted - actual))/float(len(predicted))

if __name__ == "__main__":

    training_set = []
    training_labels = []
    testing_set = []
    testing_labels = []


    # Code for loading data
    for x in xrange(0,10):
        images, labels = load_mnist(digits=[x], path='.')

        total = len(images)
        split = int(total/4)
        test_images = images[0:split][:2]
        test_labels = labels[0:split][:2]
        train_images = images[split:][:2]
        train_labels = labels[split:][:2]

        training_set.extend(train_images)
        training_labels.extend(train_labels)

        testing_set.extend(test_images)
        testing_labels.extend(test_labels)

        
    # preprocessing
    training_set = preprocess(training_set)
    training_labels = preprocess(training_labels)
    testing_set = preprocess(testing_set)
    testing_labels = preprocess(testing_labels)
    print len(testing_set)
    print testing_set.shape
    
    # pick training and testing set
    # YOU HAVE TO CHANGE THIS TO PICK DIFFERENT SET OF DATA


    #build_classifier is a function that takes in training data and outputs an sklearn classifier.
    classifier = build_classifier(training_set, training_labels)
    save_classifier(classifier, training_set, training_labels)
    classifier = pickle.load(open('classifier_1.p', 'rb'))
    predicted = classify(testing_set, classifier)
    print error_measure(predicted, testing_labels)
