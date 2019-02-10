# boxFilter.py
# John Hacker's Box Filtering
# CAP 4453 Spring 19

'''
    The 3x3 filters smooth the images and decrease the noise from the
original images. The 5x5 filters make the images more blurry and
get rid of more of the noise.
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

# Open the images
img1 = cv2.imread("./Images/image1.png", 1)
img2 = cv2.imread("./Images/image2.png", 1)

# Create the 3x3 kernel
kernel3 = numpy.ones((3,3), numpy.float32)/9

# Convolute the images with the kernel
box31 = cv2.filter2D(img1, -1, kernel3)
box32 = cv2.filter2D(img2, -1, kernel3)

# Create the 5x5 kernel
kernel5 = numpy.ones((5,5), numpy.float32)/25

# Convolute images with the kernel
box51 = cv2.filter2D(img1, -1, kernel5)
box52 = cv2.filter2D(img2, -1, kernel5)

# Plot the image1s
pyplot.subplot(231), pyplot.imshow(img1), pyplot.title('Original')
pyplot.xticks([]), pyplot.yticks([])
pyplot.subplot(232), pyplot.imshow(box31), pyplot.title('3x3 Box Filtered')
pyplot.xticks([]), pyplot.yticks([])
pyplot.subplot(233), pyplot.imshow(box51), pyplot.title('5x5 Box Filtered')
pyplot.xticks([]), pyplot.yticks([])

# Convolute the image2s
pyplot.subplot(234), pyplot.imshow(img2)
pyplot.xticks([]), pyplot.yticks([])
pyplot.subplot(235), pyplot.imshow(box32)
pyplot.xticks([]), pyplot.yticks([])
pyplot.subplot(236), pyplot.imshow(box52)
pyplot.xticks([]), pyplot.yticks([])

# Show the plotted images
pyplot.show()
