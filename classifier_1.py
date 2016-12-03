import pickle
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from mnist import load_mnist
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import random


def preprocess(images):
    #this function is suggested to help build your classifier.
    #You might want to do something with the images before
    #handing them to the classifier. Right now it does nothing.
    return np.array([i.flatten() for i in images])

def build_classifier(images, labels):
    #this will actually build the classifier. In general, it
    #will call something from sklearn to build it, and it must
    #return the output of sklearn. Right now it does nothing.
    classifier = KNeighborsClassifier(n_neighbors=5)

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
    print "entered prediction"
    return classifier.predict(images)

def error_measure(predicted, actual):
    conf_matrix = confusion_matrix(actual, predicted)
    print conf_matrix


    return sklearn.metrics.f1_score(actual, predicted, average="macro")

def handle_data(divisions=4, size=30000):
    training_set = []
    training_labels = []
    testing_set = []
    testing_labels = []

    # Code for loading data
    for x in range(10):
        images, labels = load_mnist(digits=[x], path='.')

        images = images[:size]
        labels = labels[:size]

        total = len(images)
        split = int(total / divisions)

        test_images = images[0:split]
        test_labels = labels[0:split]

        train_images =  images[split:]
        train_labels = labels[split:]

        training_set.extend(train_images)
        training_labels.extend(train_labels)

        testing_set.extend(test_images)
        testing_labels.extend(test_labels)


    # preprocessing
    raw_testing_set = testing_set
    training_set = preprocess(training_set)
    training_labels = preprocess(training_labels)
    testing_set = preprocess(testing_set)
    testing_labels = preprocess(testing_labels)

    return training_set, training_labels, testing_set, testing_labels, raw_testing_set

def save_images(predicted, actual, images):
    k = 0
    misclassified = []
    labels = []
    for x in xrange(len(images)):
        if k > 5:
            return
        elif predicted[x] != actual[x]:
            misclassified.append(images[x])
            labels.append(predicted[x])
            k += 1

    for i in xrange(len(misclassified)):
        plt.imshow(misclassified[i], cmap = 'gray')
        title = str(random.randint(0, 1000))
        plt.title('Misclassified Image As: ' + str(labels[i]))
        plt.savefig(title)


if __name__ == "__main__":

    training_set = []
    training_labels = []
    testing_set = []
    testing_labels = []
    divisions = [2,5,10]

    for d in divisions:
        training_set, training_labels, testing_set, testing_labels, raw_testing_set = handle_data(divisions=d, size=20)
        print '========================' + str(d) + ' ==================================='
        #build_classifier is a function that takes in training data and outputs an sklearn classifier.
        classifier = build_classifier(training_set, training_labels)
        save_classifier(classifier, training_set, training_labels)
        classifier = pickle.load(open('classifier_1.p', 'rb'))
        predicted = classify(testing_set, classifier)
        save_images(predicted, testing_labels, raw_testing_set)
        print error_measure(predicted, testing_labels)
