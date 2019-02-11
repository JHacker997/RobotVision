# entropy.py
# John Hacker's Entropy Thresholding
# CAP 4453 Spring 19

'''
    I create a histogram to analyze the image4. I make arrays of A and B
with the probability ratios specified in the assignment for each possible 
threshold. I use those arrays calculate the entropy of each threshold. I
use the threshold that cause the highest total entropy between A and B
to turn the greyscaleimages into black and white. As the assignment asks,
I only perform the experiment on image4 and then apply the found threshold
to each image1, image2, and image4.
'''

import PIL
import matplotlib
import numpy
import scipy
import svmutil
import cv2

from PIL import Image
from matplotlib import pyplot
from scipy import ndimage

# Returns frequency array for each intensity of an image
def createHistogram(img, maximum):
    # Create the frequency array
    intensities = numpy.zeros(256)

    # Find how many intensities need to be grouped together
    interval = 256 / maximum

    # Create array of image
    imgArray = numpy.array(img, dtype=int)

    # Find th height and width of the image
    width, height = img.size

    # Find all the intensities in the image
    for i in range(height):
        for j in range(width):
            try:
                intensities[imgArray[i][j][0]] += 1
            except IndexError:
                intensities[imgArray[i][j]] += 1

    # Group the intensities so they can be plotted into the bins
    graphIntensities = numpy.zeros(maximum)
    curInt = 0
    for i in range(256):
        if (i % interval == 0 and i != 0):
            curInt += 1
        graphIntensities[curInt] += intensities[i]
    
    # Return the intensities
    return graphIntensities

# Returns the entropy of a given system
def H(p):
    # Find the size of the system
    size = len(p)

    # Start the entropy at none
    entropy = 0

    # Perform the entropy formula
    for i in range(size):
        if (p[i] != 0 or p[i] != 1):
            entropy += p[i] * numpy.log10(p[i])

    # Return the magnitude of the entropy
    return entropy * -1

# Returns a binary image given a grayscale image and a threshold
def O(img, threshold):
    # Create array of image
    imgArr = numpy.array(img, dtype=int)

    # Find the shape of the image
    height = len(imgArr)
    width = len(imgArr[0])

    # Loop through all the pixels in the image
    for i in range(height):
        for j in range(width):
            # Set the pixel to 0 if below the image and 1 otheriwse
            try:
                if (imgArr[i][j][0] >= threshold):
                    img[i][j] = 1
                else:
                    img[i][j] = 0
            except IndexError:
                if (imgArr[i][j] >= threshold):
                    img[i][j] = 1
                else:
                    img[i][j] = 0

    return img

# Returns the threshold that results in the most entropy for a given image
def Threshold(img):
    # Find the frequency of all the intensities
    histogram = createHistogram(img, 256)

    # Find the shape of the image
    height, width = img.size
    N = height * width

    # Set the max and threshold to defaults
    maximum = 0
    T = 0

    # Loop though all the possible thresholds
    for i in range(256):
        # Create the probabilty systems
        A = numpy.ones((i + 1))
        B = numpy.ones((255 - i))

        # Set the threshold probabilty to default
        probT = 0

        # Find the size of each system
        aLen = len(A)
        bLen = len(B)

        # Fill each system and calculate the threshold probability
        for t in range(aLen):
            probT += histogram[t] / N
            A[t] = histogram[t] / N
        for t in range(bLen):
            B[t] = histogram[t + aLen] / N
        try:
            A /= probT
        except ZeroDivisionError:
            pass
        try:
            B /= (1 - probT)
        except ZeroDivisionError:
            pass

        # Find the total entropy
        entropySum = H(A) + H(B)

        # Check if new max entropy
        if (maximum < entropySum or maximum == 0):
            # Set new max entropy
            maximum = entropySum

            # Save threshold for new max entropy
            T = i

    return T

# Image paths
path4 = "./Images/image4.png"
path1 = "./Images/image1.png"
path2 = "./Images/image2.png"

# Open all the grayscale images and plot
img4P = Image.open(path4)
img4 = cv2.imread(path4, 0)
pyplot.subplot(231), pyplot.imshow(img4, cmap='gray'), pyplot.title('Gray 4')
pyplot.xticks([]), pyplot.yticks([])
img1P = Image.open(path1)
img1 = cv2.imread(path1, 0)
pyplot.subplot(232), pyplot.imshow(img1, cmap='gray'), pyplot.title('Gray 4')
pyplot.xticks([]), pyplot.yticks([])
img1P = Image.open(path2)
img2 = cv2.imread(path2, 0)
pyplot.subplot(233), pyplot.imshow(img2, cmap='gray'), pyplot.title('Gray 4')
pyplot.xticks([]), pyplot.yticks([])

# Find the entropy threshold in the image4
thresh = Threshold(img4P)

# Convert the grayscale images to binary and plot
bin4 = O(img4, thresh)
pyplot.subplot(234), pyplot.imshow(bin4, cmap='gray'), pyplot.title('Binary 4')
pyplot.xticks([]), pyplot.yticks([])
bin1 = O(img1, thresh)
pyplot.subplot(235), pyplot.imshow(bin1, cmap='gray'), pyplot.title('Binary 1')
pyplot.xticks([]), pyplot.yticks([])
bin2 = O(img2, thresh)
pyplot.subplot(236), pyplot.imshow(bin2, cmap='gray'), pyplot.title('Binary 2')
pyplot.xticks([]), pyplot.yticks([])

# Show the plotted images
pyplot.show()