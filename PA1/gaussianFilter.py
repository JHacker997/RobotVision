# gaussianFilter.py
# John Hacker's Box Filter
# CAP 4453

#   Increasing the sigma increased how similar each pixel is to each other.
# The images from this filter are significantly darker than the output
# from the previous two questions. 
#   The image1 seems to be best smoothed by the median filter and the image2
# seems to be best smoothed by the Gaussian filter. I think image1 is best 
# smooted by median filter because it has a lot of noise and the median filter
# is best at taking out outliers/noise. I think image2 is best smoothed by box 
# filter because it has blocky diagonals and the box filter takes all pixels
# around a pixel to find its value, giving increasing piority to pixels toward
# the center.

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
def gaussFilter(img, size, sigma):
    sum = 0

    # Create the kernel
    kernel = numpy.ones((size, size), numpy.float32)

    # Find the offset from the center of the kernel
    offset = (size // 2) * -1

    # Loop through all the pixles in the kernel
    for i in range(0, size):
        for j in range(0, size):
            # Find x and y for the gaussian function
            x = i + offset
            y = j + offset

            # Calculate the gaussian value for the current kernel pixel
            # Based on formula from class notes
            kernel[i][j] = (1 / (2 * numpy.pi * numpy.square(sigma))) * numpy.exp(-1 * (numpy.square(x) + numpy.square(y)) / (numpy.square(sigma)))
            sum = sum + kernel[i][j]

    print(str(kernel))
    print(str(sum))

    # Return the smoothed image
    return cv2.filter2D(img, -1, kernel)

# Open the images
img1 = cv2.imread("./Images/image1.png", 1)
img2 = cv2.imread("./Images/image2.png", 1)

# Apply the gaussian filter to the image1s with sigmas 3, 5, and 7
gauss31 = gaussFilter(img1, 3, 3)
gauss51 = gaussFilter(img1, 3, 5)
gauss71 = gaussFilter(img1, 3, 7)

# Plot the smoothed image1s
pyplot.subplot(231), pyplot.imshow(gauss31), pyplot.title('Gauss sigma=3')
pyplot.xticks([]), pyplot.yticks([])
pyplot.subplot(232), pyplot.imshow(gauss51), pyplot.title('Gauss sigma=5')
pyplot.xticks([]), pyplot.yticks([])
pyplot.subplot(233), pyplot.imshow(gauss71), pyplot.title('Gauss sigma=7')
pyplot.xticks([]), pyplot.yticks([])

# Apply the gaussian filter to the image2s with sigmas 3, 5, and 7
gauss32 = gaussFilter(img2, 3, 3)
gauss52 = gaussFilter(img2, 3, 5)
gauss72 = gaussFilter(img2, 3, 7)

# Plot the smoothed image2s
pyplot.subplot(234), pyplot.imshow(gauss32)
pyplot.xticks([]), pyplot.yticks([])
pyplot.subplot(235), pyplot.imshow(gauss52)
pyplot.xticks([]), pyplot.yticks([])
pyplot.subplot(236), pyplot.imshow(gauss72)
pyplot.xticks([]), pyplot.yticks([])

# Show the plotted images
pyplot.show()