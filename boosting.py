# import packages (including scikit-learn packages)
from sklearn.ensemble import AdaBoostClassifier # Use this function for adaboosting
from sklearn.linear_model import SGDClassifier
from mnist import load_mnist
from sklearn.metrics import confusion_matrix, f1_score
import numpy as np
from sklearn import svm
#Partners: Daniel Nussbaum and Sonia Nigam

def boosting_A(training_set, training_labels, testing_set, testing_labels):
	'''
	Input Parameters:
		- training_set: a 2-D numpy array that contains training examples (size: the number of training examples X the number of attributes)
		  (NOTE: If a training example is 10x10 images, the number of attributes will be 100. You need to reshape your training example)
		- training_labels: a 1-D numpy array that labels of training examples (size: the number of training examples)
		- testing_set: a 2-D numpy array that contains testing examples  (size: the number of testing examples X the number of attributes)
		- testing_labels: a 1-D numpy array that labels of testing examples (size: the number of testing examples)

	Returns:
		- predicted_labels: a 1-D numpy array that contains the labels predicted by the classifier. Labels in this array should be sorted in the same order as testing_labels
		- confusion_matrix: a 2-D numpy array of confusion matrix (size: the number of classes X the number of classes)
	'''

	# Build boosting algorithm for question 6-A
	classifier = AdaBoostClassifier()

	# reshape the training data
	training_labels.shape = training_labels.shape[0]

	# fit the data to the classifier
	classifier.fit(training_set, training_labels)

	# make predictions
	predicted_labels = classifier.predict(testing_set)

	# get and print a confusion matrix and f1-score
	conf_matrix = confusion_matrix(testing_labels, predicted_labels)
	print conf_matrix
	print "F MEASURE: " + str(f1_score(testing_labels, predicted_labels, average="macro"))
	return predicted_labels, conf_matrix

def boosting_B(training_set, training_labels, testing_set, testing_labels):
	'''
	Input Parameters:
		- training_set: a 2-D numpy array that contains training examples (size: the number of training examples X the number of attributes)
		(NOTE: If a training example is 10x10 images, the number of attributes will be 100. You need to reshape your training example)
		- training_labels: a 1-D numpy array that labels of training examples (size: the number of training examples)
		- testing_set: a 2-D numpy array that contains testing examples  (size: the number of testing examples X the number of attributes)
		- testing_labels: a 1-D numpy array that labels of testing examples (size: the number of testing examples)

	Returns:
		- predicted_labels: a 1-D numpy array that contains the labels predicted by the classifier. Labels in this array should be sorted in the same order as testing_labels
		- confusion_matrix: a 2-D numpy array of confusion matrix (size: the number of classes X the number of classes)
	'''
	# Build boosting algorithm for question 6-B
	# set classifier
	classifier = AdaBoostClassifier(svm.LinearSVC(), algorithm='SAMME')

	# changing shape
	training_labels.shape = training_labels.shape[0]

	# fitting classifer
	classifier.fit(training_set, training_labels)

	# finding predictions
	predicted_labels = classifier.predict(testing_set)

	# getting confusion matrix and f1-measure
	conf_matrix = confusion_matrix(testing_labels, predicted_labels)
	print conf_matrix
	print "F MEASURE: " + str(f1_score(testing_labels, predicted_labels, average="macro"))
	return predicted_labels, conf_matrix

def preprocess(images):
	#this function is suggested to help build your classifier.
	#You might want to do something with the images before
	#handing them to the classifier. Right now it does nothing.
	return np.array([i.flatten() for i in images])

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
    training_set = preprocess(training_set)
    training_labels = preprocess(training_labels)
    testing_set = preprocess(testing_set)
    testing_labels = preprocess(testing_labels)

    return training_set, training_labels, testing_set, testing_labels


def main():
	"""
	This function runs boosting_A() and boosting_B() for problem 7.
	Load data set and perform adaboosting using boosting_A() and boosting_B()
	"""
	training_set, training_labels, testing_set, testing_labels = handle_data(15000)

	print "########### BOOSTING A ############"
	predicted_labels, confusion_matrix = boosting_A(
		training_set,
		training_labels,
		testing_set,
		testing_labels
	)

	print "########### BOOSTING B ############"
	predicted_labels, confusion_matrix = boosting_B(
		training_set,
		training_labels,
		testing_set,
		testing_labels
	)




if __name__ == '__main__':
	main()
