# histogram.py
# John Hacker's Histogram
# CAP 4453 Spring 19

#   This program runs by showing the histogram with 256 bins, then shows
# the histogram with 128 bins AFTER THE PREVIOUS PLOT IS CLOSED. Then the
# histogram for 64 bins will show after the previous plot is closed.
#   Histograms are made by using a bar graph to plot the number of pixels
# that have an intensity equal to the x-axis value or within a bin that
# starts with that value.

import PIL
import matplotlib
import numpy
import scipy
import svmutil
import cv2

from PIL import Image
from matplotlib import pyplot

def createHistogram(img, maximum):
    # Create the frequency array
    intensities = numpy.zeros(256)

    # Create array to organize bins
    bins = numpy.zeros(maximum)

    # Find how many intensities need to be grouped together
    interval = 256 / maximum

    # Create array of image
    imgArray = numpy.array(img, dtype=int)

    # Find th height and width of the image
    width, height = img.size

    # Find all the intensities in the image
    for i in range(height):
        for j in range(width):
            intensities[imgArray[i][j][0]] += 1

    # Group the intensities so they can be plotted into the bins
    graphIntensities = numpy.zeros(maximum)
    curInt = 0
    for i in range(256):
        if i % interval == 0 and i is not 0:
            curInt += 1
        graphIntensities[curInt] += intensities[i]
    
    # Value each bin the same as its index
    for i in range(maximum):
        bins[i] = i
    
    # Plot the intensities into a bar graph
    pyplot.bar(bins, graphIntensities)
    pyplot.show()

    return

# Open the image4
img = Image.open("./Images/image4.png")

# Create histograms for intervals of 256, 128, 64
createHistogram(img, 256)
createHistogram(img, 128)
createHistogram(img, 64)
