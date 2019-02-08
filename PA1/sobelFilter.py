# sobelFilter.py
# John Hacker's Sobel Filtering
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

def sobelX(img):
    # Create the kernel
    kernel = numpy.ones((3, 3), numpy.float32)
    kernel[0][1] = 0
    kernel[0][2] = -1
    kernel[1][0] = 2
    kernel[1][1] = 0
    kernel[1][2] = -2
    kernel[2][1] = 0
    kernel[2][2] = -1

    # Return the smoothed image
    return cv2.filter2D(img, -1, kernel)

def sobelY(img):
    # Create the kernel
    kernel = numpy.ones((3, 3), numpy.float32)
    kernel[0][1] = 2
    kernel[1][0] = 0
    kernel[1][1] = 0
    kernel[1][2] = 0
    kernel[2][0] = -1
    kernel[2][1] = -2
    kernel[2][2] = -1

    # Return the smoothed image
    return cv2.filter2D(img, -1, kernel)

def magnitude(imgX, imgY):
    width = imgX.shape[0]
    height = imgX.shape[1]

    print(str(imgX))
    print(str(imgY))


    imgMag = numpy.ones((width, height), numpy.float32)

    # Loop through all the pixels in the kernel
    for i in range(0, width):
        for j in range(0, height):
            # Make the current pixel the magnitude of the other two images
            imgMag[i][j] = numpy.sqrt(numpy.square(imgX[i][j]) + numpy.square(imgY[i][j]))

    return imgMag

img1 = Image.open("./Images/image1.png")
img2 = Image.open("./Images/image2.png")

sobx1 = sobelX(img1)
soby1 = sobelY(img1)
sob1 = Image.fromarray(magnitude(sobx1, soby1))

pyplot.subplot(121), pyplot.imshow(sob1), pyplot.title('Sobel Image 1')
pyplot.xticks([]), pyplot.yticks([])