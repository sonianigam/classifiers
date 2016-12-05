import matplotlib.pyplot as plt
import numpy as np
from mnist import load_mnist

################# Problem 1A: 50 samples of digit 8 ##################

# only returns images of digit 8
images, labels = load_mnist(digits=[8], path='.')

#shows 50 images of the digit 8
for i in xrange(50):
	plt.imshow(images[i], cmap = 'gray')
	plt.title('Handwritten image of the digit ' + str(labels[i]))
	plt.show()

################### Problem 1B #####################
total = 0

#returns length of image set corresponding to each digit
for x in xrange(0,10):
	images, labels = load_mnist(digits=[x], path='.')
	print "The total number of images for digit " + str(x) + ": "+ str(len(images))
	total += len(images)

#prints total number of images
print "The total number of images: " + str(total)