# medianFilter.py
# John Hacker's Median Filtering
# CAP 4453 Spring 19

#   As the kernel increases in size, the angles get more smooth
# and blurry and the corners become more rounded.

import PIL
import matplotlib
import numpy
import scipy
import svmutil
import cv2

from PIL import Image
from matplotlib import pyplot

# Input: image and value for a size x size kernel
# Output: smoothed image with median filter applied
def medianFilter(img, size):
    # Find the width and height of the origianl image
    width1 = img.width
    height1 = img.height

    # Create a new image that is a copy of the original image
    medImg = Image.new("L", (width1, height1), "white")

    # Create the kernel
    kernelSize = numpy.square(size)
    pixels = [(0, 0)] * kernelSize

    # Find how far the counters should go to reach the edges of th kernel
    minOff = (size // 2) * -1
    maxOff = minOff * -1

    # Loop through all the pixels in the original image
    for i in range(1, int(width1 + minOff)):
        for j in range(1, int(height1 + minOff)):
            # Create the counters to go through the kernel
            m = minOff
            n = minOff

            # Loop through all the pixels in the kernel
            for k in range(0, kernelSize):
                pixels[k] = img.getpixel((i + m, j + n))
                
                # Increment the kernel counters (n first, then m when n is max)
                if (n == maxOff):
                    n = minOff
                    m = m + 1
                else:
                    n = n + 1
            
            # Sort the pixels
            pixels.sort()

            # Replace the current pixel with the median value in its kernel
            medImg.putpixel((i, j), (pixels[kernelSize // 2]))
    
    # Return the smoothed image
    return medImg

# Open the first image and smooth it to each kernel size
img1 = Image.open("./Images/image1.png")
med31 = medianFilter(img1, 3)
med51 = medianFilter(img1, 5)
med71 = medianFilter(img1, 7)

# Plot all of the smoothed image1s
pyplot.subplot(231), pyplot.imshow(med31), pyplot.title('3x3 Median Filter')
pyplot.xticks([]), pyplot.yticks([])
pyplot.subplot(232), pyplot.imshow(med51), pyplot.title('5x5 Median Filter')
pyplot.xticks([]), pyplot.yticks([])
pyplot.subplot(233), pyplot.imshow(med71), pyplot.title('7x7 Median Filter')
pyplot.xticks([]), pyplot.yticks([])

# Open the second image and smooth it to each kernel size
img2 = Image.open("./Images/image2.png")
med32 = medianFilter(img2, 3)
med52 = medianFilter(img2, 5)
med72 = medianFilter(img2, 7)

# Plot all the smoothed image2s
pyplot.subplot(234), pyplot.imshow(med32), pyplot.title('3x3 Median Filter')
pyplot.xticks([]), pyplot.yticks([])
pyplot.subplot(235), pyplot.imshow(med52), pyplot.title('5x5 Median Filter')
pyplot.xticks([]), pyplot.yticks([])
pyplot.subplot(236), pyplot.imshow(med72), pyplot.title('7x7 Median Filter')
pyplot.xticks([]), pyplot.yticks([])

# Show all the plotted images
pyplot.show()
