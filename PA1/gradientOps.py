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

from PIL import Image
from matplotlib import pyplot

# Input: image, kenerl size, sigma value
# Output: image with gaussian filter applied
def gradientY(img, size):
    # Create the kernel
    kernel = numpy.ones((size, size), numpy.float32)

    # Loop through all the pixles in the kernel
    for i in range(0, size):
        for j in range(0, size):
            if (i == 0):
                kernel[i][j] = 1
            if (i == 1):
                kernel[i][j] = 0
            if (i == 2):
                kernel[i][j] = -1
    
    print(str(kernel))

    # Return the smoothed image
    return cv2.filter2D(img, -1, kernel)

def gradientX(img, size):
    # Create the kernel
    kernel = numpy.ones((size, size), numpy.float32)

    # Loop through all the pixles in the kernel
    for i in range(0, size):
        for j in range(0, size):
            if (j == 0):
                kernel[i][j] = 1
            if (j == 1):
                kernel[i][j] = 0
            if (j == 2):
                kernel[i][j] = -1
    
    print(str(kernel))

    # Return the smoothed image
    return cv2.filter2D(img, -1, kernel)

def gradientMag(imgX, imgY):
    width = imgX.shape[0]
    height = imgX.shape[1]

    imgMag = numpy.ones((width, height), numpy.float32)
    # Loop through all the pixles in the kernel
    for i in range(0, width):
        for j in range(0, height):
            for k in range(0, 3):
                imgMag[i][j][k] = numpy.sqrt(numpy.square(imgX[i][j][k]) + numpy.square(imgY[i][j][k]))
    return imgMag

# Open the images
img = cv2.imread("./Images/image3.png", 1)

# Apply the gaussian filter to the image1s with sigmas 3, 5, and 7
gradY = gradientY(img, 3)

# Plot the smoothed image1s
pyplot.subplot(131), pyplot.imshow(gradY), pyplot.title('Y Gradient')
pyplot.xticks([]), pyplot.yticks([])

# Apply the gaussian filter to the image2s with sigmas 3, 5, and 7
gradX = gradientX(img, 3)

# Plot the smoothed image2s
pyplot.subplot(132), pyplot.imshow(gradX), pyplot.title('X Gradient')
pyplot.xticks([]), pyplot.yticks([])

# Apply the gaussian filter to the image2s with sigmas 3, 5, and 7
#gradMag = Image.fromarray(gradientMag(gradX, gradY))

# Plot the smoothed image2s
#pyplot.subplot(133), pyplot.imshow(gradMag), pyplot.title('Magnitude Gradient')
#pyplot.xticks([]), pyplot.yticks([])

# Show the plotted images
pyplot.show()