# sobelFilter.py
# John Hacker's Sobel Filtering
# CAP 4453 Spring 19

'''
    The Sobel Filtering shows where edges in the image are. The firt
image has a lot of noise, so the Sobel filter finds a lot of edges
around all the random dots along with the shapes. It struggles to
find the edges in the blocky gradient in the first image that it
can find in the second image. The second image finds the edges very
cleanly because it did not have a lot of noise to begin with.
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

# Returns an image with the Sobel filter applied to a given image
def sobel(img):
    # Create and array of the image
    imgArray = numpy.array(img, dtype=float)

    # Create the kernel
    kernelX = numpy.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kernelY = numpy.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    # Return the smoothed image
    sobX = ndimage.convolve(imgArray, kernelX)
    sobY = ndimage.convolve(imgArray, kernelY)

    # Find the size of the image
    width = len(imgArray)
    height = len(imgArray[0])

    # Create the array for the sobel filtered image
    sob = numpy.ndarray((width, height))

    # Fill the image with the magnitude of the x and y sobel filters
    for i in range(height):
        for j in range(width):
            sob[i][j] = numpy.sqrt(numpy.square(sobX[i][j]) + numpy.square(sobY[i][j]))
    
    # Turn the array into an image
    return Image.fromarray(sob)

# Open the images
img1 = Image.open("./Images/image1.png")
img2 = Image.open("./Images/image2.png")

# Use the sobel filter on the images
sob1 = sobel(img1)
sob2 = sobel(img2)

# Plot the original and filtered images
pyplot.subplot(221), pyplot.imshow(img1, cmap = 'gray'), pyplot.title('Original Image 1')
pyplot.xticks([]), pyplot.yticks([])
pyplot.subplot(222), pyplot.imshow(sob1, cmap = 'gray'), pyplot.title('Sobel Image 1')
pyplot.xticks([]), pyplot.yticks([])

pyplot.subplot(223), pyplot.imshow(img2, cmap = 'gray'), pyplot.title('Original Image 1')
pyplot.xticks([]), pyplot.yticks([])
pyplot.subplot(224), pyplot.imshow(sob2, cmap = 'gray'), pyplot.title('Sobel Image 1')
pyplot.xticks([]), pyplot.yticks([])

# Show the plotted images
pyplot.show()