# gradientOps.py
# John Hacker's Gradient Operations
# CAP 4453 Spring 19

'''
    I displayed the results of backward, forward, and central finite
difference in the x and y axes and then showed the magnitude of the RGB
values between the x and y gradients for each direction.
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

# Input: image
# Output: image with backward finite differene applied
def backwardX(img):
    # Create the kernel
    kernel = numpy.array([[-1, 1]]) / 2

    # Return the smoothed image
    return cv2.filter2D(img, -1, kernel)

# Input: image
# Output: image with forward finite differene applied
def forwardX(img):
    # Create the kernel
    kernel = numpy.array([[1, -1]]) / 2

    # Return the smoothed image
    return cv2.filter2D(img, -1, kernel)

# Input: image
# Output: image with central finite differene applied
def centralX(img):
    # Create the kernel
    kernel = numpy.array([[1, 0, -1]]) / 2

    # Return the smoothed image
    return cv2.filter2D(img, -1, kernel)


# Input: image
# Output: image with backward finite differene applied
def backwardY(img):
    # Create the kernel
    kernel = numpy.array([[1], [-1]]) / 2

    # Return the smoothed image
    return cv2.filter2D(img, -1, kernel)

# Input: image
# Output: image with forward finite differene applied
def forwardY(img):
    # Create the kernel
    kernel = numpy.array([[-1], [1]]) / 2

    # Return the smoothed image
    return cv2.filter2D(img, -1, kernel)

# Input: image
# Output: image with central finite differene applied
def centralY(img):
    # Create the kernel
    kernel = numpy.array([[-1], [0], [1]]) / 2

    # Return the smoothed image
    return cv2.filter2D(img, -1, kernel)


# Returns an image that is the magnitde of two given images
def gradientMag(imgX, imgY):
    # Find the size of the image
    height = imgX.shape[0]
    width = imgX.shape[1]

    #imgMag = numpy.ones((width, height, 3), numpy.float32)
    imgMag = numpy.ndarray((height, width, 3))

    # Loop through all the pixels in the kernel
    for i in range(0, height):
        for j in range(0, width):
            for k in range(0, 3):
                # Make the current pixel the magnitude of the other two images
                imgMag[i][j][k] = int(numpy.sqrt((numpy.square(int(imgX[i][j][k]))) + (numpy.square(int(imgY[i][j][k])))))

    imgMag = imgMag.astype(int)

    return imgMag

# Open the images
img = cv2.imread("./Images/image3.png")

# Apply the gaussian filter to the image2s with sigmas 3, 5, and 7
backX = backwardX(img)
backY = backwardY(img)

# Find the magnitudes of the two forward gradient images
backMag = gradientMag(backX, backY)

# Plot the forward gradients of image3
pyplot.subplot(331), pyplot.imshow(backX), pyplot.title('X Backward')
pyplot.xticks([]), pyplot.yticks([])
pyplot.subplot(332), pyplot.imshow(backY), pyplot.title('Y Backward')
pyplot.xticks([]), pyplot.yticks([])
pyplot.subplot(333), pyplot.imshow(backMag), pyplot.title('Magnitude Backward')
pyplot.xticks([]), pyplot.yticks([])

# Apply the gaussian filter to the image2s with sigmas 3, 5, and 7
forX = forwardX(img)
forY = forwardY(img)

# Find the magnitudes of the two forward gradient images
forMag = gradientMag(forX, forY)

# Plot the forward gradients of image3
pyplot.subplot(334), pyplot.imshow(forX), pyplot.title('X Forward')
pyplot.xticks([]), pyplot.yticks([])
pyplot.subplot(335), pyplot.imshow(forY), pyplot.title('Y Forward')
pyplot.xticks([]), pyplot.yticks([])
pyplot.subplot(336), pyplot.imshow(forMag), pyplot.title('Magnitude Forward')
pyplot.xticks([]), pyplot.yticks([])

# Apply the gaussian filter to the image2s with sigmas 3, 5, and 7
centX = centralX(img)
centY = centralY(img)

# Find the magnitudes of the two forward gradient images
centMag = gradientMag(centX, centY)

# Plot the forward gradients of image3
pyplot.subplot(337), pyplot.imshow(centX), pyplot.title('X Central')
pyplot.xticks([]), pyplot.yticks([])
pyplot.subplot(338), pyplot.imshow(centY), pyplot.title('Y Central')
pyplot.xticks([]), pyplot.yticks([])
pyplot.subplot(339), pyplot.imshow(centMag), pyplot.title('Magnitude Central')
pyplot.xticks([]), pyplot.yticks([])

# Show the plotted images
pyplot.show()