# gradientOps.py
# John Hacker's Gradient Operations
# CAP 4453 Spring 19

# 
# 
# 
# 

import PIL
import matplotlib
import numpy
import scipy
import svmutil
import cv2
import math

from PIL import Image
from matplotlib import pyplot

# Input: image, kenerl size
# Output: image with backward finite differene applied
def backwardX(img, size):
    # Create the kernel
    kernel = numpy.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]) / 3
    #kernel = numpy.array([[-1, 1]])

    # Return the smoothed image
    return cv2.filter2D(img, -1, kernel)

# Input: image, kenerl size
# Output: image with forward finite differene applied
def forwardX(img, size):
    # Create the kernel
    kernel = numpy.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]]) / 3
    #kernel = numpy.array([[1, -1]])

    # Return the smoothed image
    return cv2.filter2D(img, -1, kernel)


# Input: image, kenerl size
# Output: image with backward finite differene applied
def backwardY(img, size):
    # Create the kernel
    kernel = numpy.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]]) / 3
    #kernel = numpy.array([[-1], [1]])

    # Return the smoothed image
    return cv2.filter2D(img, -1, kernel)

# Input: image, kenerl size
# Output: image with forward finite differene applied
def forwardY(img, size):
    # Create the kernel
    kernel = numpy.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]) / 3
    #kernel = numpy.array([[1], [-1]])

    # Return the smoothed image
    return cv2.filter2D(img, -1, kernel)


def gradientMag(imgX, imgY):
    height = imgX.shape[0]
    width = imgX.shape[1]

    #imgMag = numpy.ones((width, height, 3), numpy.float32)
    imgMag = numpy.ndarray((height, width, 3))
    imgMag.astype(int)

    # Loop through all the pixels in the kernel
    for i in range(0, height):
        for j in range(0, width):
            for k in range(0, 3):
                # Make the current pixel the magnitude of the other two images
                imgMag[i][j][k] = int(math.sqrt((imgX[i][j][k]**2) + (imgY[i][j][k]**2)))
                #imgMag[i][j][k] = (imgX[i][j][k] + imgY[i][j][k]) / 2

    imgMag = imgMag.astype(int)

    return imgMag

# Open the images
img = cv2.imread("./Images/image3.png")

# Apply the gaussian filter to the image2s with sigmas 3, 5, and 7
backX = backwardX(img, 3)
backY = backwardY(img, 3)

# Find the magnitudes of the two forward gradient images
backMag = gradientMag(backX, backY)

# Plot the forward gradients of image3
pyplot.subplot(231), pyplot.imshow(backX), pyplot.title('X Backward')
pyplot.xticks([]), pyplot.yticks([])
pyplot.subplot(232), pyplot.imshow(backY), pyplot.title('Y Backward')
pyplot.xticks([]), pyplot.yticks([])
pyplot.subplot(233), pyplot.imshow(backMag), pyplot.title('Magnitude Backward')
pyplot.xticks([]), pyplot.yticks([])

# Apply the gaussian filter to the image2s with sigmas 3, 5, and 7
forX = forwardX(img, 3)
forY = forwardY(img, 3)

# Find the magnitudes of the two forward gradient images
forMag = gradientMag(forX, forY)

# Plot the forward gradients of image3
pyplot.subplot(234), pyplot.imshow(forX), pyplot.title('X Forward')
pyplot.xticks([]), pyplot.yticks([])
pyplot.subplot(235), pyplot.imshow(forY), pyplot.title('Y Forward')
pyplot.xticks([]), pyplot.yticks([])
pyplot.subplot(236), pyplot.imshow(forMag), pyplot.title('Magnitude Forward')
pyplot.xticks([]), pyplot.yticks([])

# Show the plotted images
pyplot.show()