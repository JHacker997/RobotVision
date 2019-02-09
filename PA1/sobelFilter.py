# sobelFilter.py
# John Hacker's Sobel Filtering
# CAP 4453 Spring 19

#   The Sobel Filtering shows where edges in the image are. The firt
# image has a lot of noise, so the Sobel filter finds a lot of edges
# around all the random dots along with the shapes. It struggles to
# find the edges in the blocky gradient in the first image that it
# can find in the second image. The second image finds the edges very
# cleanly because it did not have a lot of noise to begin with.

import PIL
import matplotlib
import numpy
import scipy
import svmutil
import cv2

from PIL import Image
from matplotlib import pyplot
from scipy import ndimage

def sobel(img):
    imgArray = numpy.array(img, dtype=float)

    # Create the kernel
    kernelX = numpy.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kernelY = numpy.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    # Return the smoothed image
    sobX = ndimage.convolve(imgArray, kernelX)
    sobY = ndimage.convolve(imgArray, kernelY)

    width = len(imgArray)
    height = len(imgArray[0])

    sob = numpy.ndarray((width, height))

    for i in range(height):
        for j in range(width):
            sob[i][j] = numpy.sqrt(numpy.square(sobX[i][j]) + numpy.square(sobY[i][j]))
    
    return Image.fromarray(sob)
    #return cv2.filter2D(img, -1, kernel)


def magnitude(imgX, imgY):
    width = imgX.shape[0]
    height = imgX.shape[1]

    #imgMag = numpy.ones((width, height), numpy.float32)
    imgMag = numpy.ndarray((height, width))


    # Loop through all the pixels in the kernel
    for i in range(0, width):
        for j in range(0, height):
            # Make the current pixel the magnitude of the other two images
            imgMag[i][j] = numpy.sqrt(int(numpy.square(imgX[i][j])) + numpy.square(int(imgY[i][j])))

    imgMag.astype(int)

    return imgMag

img1 = Image.open("./Images/image1.png")
img2 = Image.open("./Images/image2.png")

sobx1 = sobel(img1)

sobx2 = sobel(img2)

pyplot.subplot(221), pyplot.imshow(img1, cmap = 'gray'), pyplot.title('Original Image 1')
pyplot.xticks([]), pyplot.yticks([])
pyplot.subplot(222), pyplot.imshow(sobx1, cmap = 'gray'), pyplot.title('Sobel Image 1')
pyplot.xticks([]), pyplot.yticks([])

pyplot.subplot(223), pyplot.imshow(img2, cmap = 'gray'), pyplot.title('Original Image 1')
pyplot.xticks([]), pyplot.yticks([])
pyplot.subplot(224), pyplot.imshow(sobx2, cmap = 'gray'), pyplot.title('Sobel Image 1')
pyplot.xticks([]), pyplot.yticks([])

pyplot.show()