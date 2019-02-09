# fastGaussian.py
# John Hacker's Fast Gaussian Filtering
# CAP 4453 Spring 19

#   Fast Gaussian's runtime is O(w+h) where the normal Gaussian Filter
# has a runtime of O(wh) where w and h are the width and height of the
# image. This means that the normal Gaussian will get a lot slower than
# Fast Gaussian when images get really big. They have basically the
# same result; so, Fast Gaussian is definitely more efficient than the
# normal Gaussian filter.

import PIL
import matplotlib
import numpy
import scipy
import svmutil
import cv2

from PIL import Image
from matplotlib import pyplot
from scipy import ndimage

# Input: image, kenerl size, sigma value
# Output: image with gaussian filter applied
def gaussX(img, sigma):
    sum = 0

    # Find the size of the kernel
    size = sigma * 6

    if (size % 2 == 0):
        size = size + 1
    print(str(size))

    # Create the kernel
    kernel = numpy.ones((1,size))

    # Find the offset from the center of the kernel
    offset = (size // 2) * -1

    # Loop through all the pixles in the kernel
    for i in range(0, size):
        # Find x and y for the gaussian function
        x = i + offset

        # Calculate the gaussian value for the current kernel pixel
        # Based on formula from class notes except for the 2 in the denominator of the exponential
        kernel[0][i] = (1 / (numpy.sqrt(2 * numpy.pi) * sigma)) * numpy.exp(-1 * (numpy.square(x)) / (2 * numpy.square(sigma)))

    # Return the smoothed image
    return ndimage.convolve(img, kernel)

# Input: image, kenerl size, sigma value
# Output: image with gaussian filter applied
def gaussY(img, sigma):
    sum = 0

    # Find the size of the kernel
    size = sigma * 6

    if (size % 2 == 0):
        size = size + 1
    print(str(size))
    # Create the kernel
    kernel = numpy.ones((size, 1), numpy.float32)

    # Find the offset from the center of the kernel
    offset = (size // 2) * -1

    # Loop through all the pixles in the kernel
    for i in range(0, size):
        # Find x and y for the gaussian function
        y = i + offset

        # Calculate the gaussian value for the current kernel pixel
        # Based on formula from class notes except for the 2 in the denominator of the exponential
        kernel[i][0] = (1 / (numpy.sqrt(2 * numpy.pi) * sigma)) * numpy.exp(-1 * (numpy.square(y)) / (2 * numpy.square(sigma)))

    # Return the smoothed image
    return ndimage.convolve(img, kernel)

# Open the images
img1 = Image.open("./Images/image1.png")
img2 = Image.open("./Images/image2.png")

img1Arr = numpy.array(img1, dtype=float)
img2Arr = numpy.array(img2, dtype=float)

# Apply the gaussian filter to the image1s with sigmas 3, 5, and 10
gauss31x = gaussX(img1Arr, 3)
gauss31 = Image.fromarray(gaussY(gauss31x, 3))
gauss51x = gaussX(img1Arr, 5)
gauss51 = Image.fromarray(gaussY(gauss51x, 5))
gauss101x = gaussX(img1Arr, 10)
gauss101 = Image.fromarray(gaussY(gauss101x, 10))

# Plot the smoothed image1s
pyplot.subplot(231), pyplot.imshow(gauss31, cmap='gray'), pyplot.title('Gauss sigma=3')
pyplot.xticks([]), pyplot.yticks([])
pyplot.subplot(232), pyplot.imshow(gauss51, cmap='gray'), pyplot.title('Gauss sigma=5')
pyplot.xticks([]), pyplot.yticks([])
pyplot.subplot(233), pyplot.imshow(gauss101, cmap='gray'), pyplot.title('Gauss sigma=10')
pyplot.xticks([]), pyplot.yticks([])

# Apply the gaussian filter to the image2s with sigmas 3, 5, and 10
gauss32x = gaussX(img2Arr, 3)
gauss32 = Image.fromarray(gaussY(gauss32x, 3))
gauss52x = gaussX(img2Arr, 5)
gauss52 = Image.fromarray(gaussY(gauss52x, 5))
gauss102x = gaussX(img2Arr, 10)
gauss102 = Image.fromarray(gaussY(gauss102x, 10))

# Plot the smoothed image2s
pyplot.subplot(234), pyplot.imshow(gauss32, cmap='gray')
pyplot.xticks([]), pyplot.yticks([])
pyplot.subplot(235), pyplot.imshow(gauss52, cmap='gray')
pyplot.xticks([]), pyplot.yticks([])
pyplot.subplot(236), pyplot.imshow(gauss102, cmap='gray')
pyplot.xticks([]), pyplot.yticks([])

# Show the plotted images
pyplot.show()