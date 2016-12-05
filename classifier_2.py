import pickle
import sklearn
from sklearn import svm
from mnist import load_mnist
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import random
#Partners: Daniel Nussbaum and Sonia Nigam


def preprocess(images):
    #this function is suggested to help build your classifier.
    #You might want to do something with the images before
    #handing them to the classifier. Right now it does nothing.
    return np.array([i.flatten() for i in images])

def build_classifier(images, labels, kernel, c):
    #this will actually build the classifier. In general, it
    #will call something from sklearn to build it, and it must
    #return the output of sklearn. Right now it does nothing.
    classifier = svm.SVC(C=c, kernel=kernel)

    # reshape the data
    labels.shape = labels.shape[0]

    # fit the classifier
    classifier.fit(images, labels)

    return classifier

##the functions below are required
def save_classifier(classifier, training_set, training_labels):
    #this saves the classifier to a file "classifier" that we will
    #load from. It also saves the data that the classifier was trained on.
    import pickle
    pickle.dump(classifier, open('classifier_2.p', 'wb'))



def classify(images, classifier):
    #runs the classifier on a set of images.
    return classifier.predict(images)

def error_measure(predicted, actual):
    #returns orginal error measure as provided in starter code
    return np.count_nonzero(abs(predicted - actual))/float(len(predicted))

def exp_error_measure(predicted, actual):
    # make a confusion matrix
    conf_matrix = confusion_matrix(actual, predicted)
    print conf_matrix
    #returns F1-meausure for experiments
    return sklearn.metrics.f1_score(actual, predicted, average="macro")


def handle_data(training_size):
    # loads the mnist data and assembles a training set (with labels) of size
    # training_size. Also gets a testing set.

    training_set = []
    training_labels = []
    testing_set = []
    testing_labels = []

    # For each digit
    for x in range(10):
        images, labels = load_mnist(digits=[x], path='.')
        training_index = int(training_size/10)

        # get the testing set from the end
        # always take the last 20 as testing data
        test_images = images[len(images)-20:]
        test_labels = labels[len(images)-20:]

        # take the indicated training set size
        train_images =  images[0:training_index]
        train_labels = labels[0:training_index]

        # build the lists
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

def save_images(predicted, actual, images, d):
    k = 0
    misclassified = []
    labels = []

    # for all images
    for x in xrange(len(images)):
        # break if you already have 5 misclassified images saved
        if k > 5:
            break
        # if they aren't the same, append the image and the misclassified label
        elif predicted[x] != actual[x]:
            misclassified.append(images[x])
            labels.append(predicted[x])
            k += 1

    # save the images
    for i in xrange(len(misclassified)):
        plt.imshow(misclassified[i], cmap = 'gray')
        title = str(random.randint(0, 1000))
        plt.title('Misclassified Image As: ' + str(labels[i]))
        plt.savefig(str(d) + "_" + title)

def experiment_one():
    # varys the size of the training set

    sizes = [500, 1000, 5000,10000, 15000]

    for s in sizes:
        training_set = []
        training_labels = []
        testing_set = []
        testing_labels = []

        # get the data
        training_set, training_labels, testing_set, testing_labels, raw_testing_set = handle_data(training_size=s)
        print '========================' + str(s) + ' ==================================='

        # build_classifier is a function that takes in training data and outputs an sklearn classifier.
        classifier = build_classifier(training_set, training_labels)

        # save the classifier
        save_classifier(classifier, training_set, training_labels)
        classifier = pickle.load(open('classifier_2.p', 'rb'))

        # predict testing data
        predicted = classify(testing_set, classifier)
        #save misclassified images
        save_images(predicted, testing_labels, raw_testing_set, s)
        # print the f-measure and conf matrix
        print exp_error_measure(predicted, testing_labels)

def experiment_two():
    # test C-values and kernels in a grid search
    Cvals = [1, 5, 10, 20, 30, 40, 50, 60]
    kernels = ["linear", "poly", "rbf", "sigmoid"]

    for c in Cvals:
        for k in kernels:
            training_set = []
            training_labels = []
            testing_set = []
            testing_labels = []

            # get the data
            training_set, training_labels, testing_set, testing_labels, raw_testing_set = handle_data(training_size=15000)
            print '======================== C:' + str(c) + ', Kernel: ' + str(k) + ' ==================================='
            #build_classifier is a function that takes in training data and outputs an sklearn classifier.
            classifier = build_classifier(training_set, training_labels, c=c, kernel=k)

            # save the classifier
            save_classifier(classifier, training_set, training_labels)
            classifier = pickle.load(open('classifier_2.p', 'rb'))
            # predict testing data
            predicted = classify(testing_set, classifier)
            #save misclassified images
            save_images(predicted, testing_labels, raw_testing_set, n)
            # print the f1 measure and conf matrix
            print exp_error_measure(predicted, testing_labels)


if __name__ == "__main__":
    training_set = []
    training_labels = []
    testing_set = []
    testing_labels = []

    # get the data and build the classifier
    training_set, training_labels, testing_set, testing_labels, raw_testing_set = handle_data(training_size=15000)
    classifier = build_classifier(training_set, training_labels, kernel="rbf", c=40)

    # save the classifier
    save_classifier(classifier, training_set, training_labels)
    classifier = pickle.load(open('classifier_2.p', 'rb'))

    # make classifications and print the error measure
    predicted = classify(testing_set, classifier)
    print error_measure(predicted, testing_labels)
