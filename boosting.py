# import packages (including scikit-learn packages)
from sklearn.ensemble import AdaBoostClassifier # Use this function for adaboosting

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
 

    return predicted_labels, confusion_matrix

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


    return predicted_labels, confusion_matrix

def main():
    """
    This function runs boosting_A() and boosting_B() for problem 7.
    Load data set and perform adaboosting using boosting_A() and boosting_B()
    """



if __name__ == '__main__':
    main()
