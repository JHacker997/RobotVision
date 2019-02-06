# boxFilter.py
# John Hacker's Box Filter
# CAP 4453

import PIL
import matplotlib
import numpy
import scipy
import svmutil
import cv2

from PIL import Image
from matplotlib import pyplot

img1 = cv2.imread("./Images/image1.png", 1)
img2 = cv2.imread("./Images/image2.png", 1)

kernel3 = numpy.ones((3,3), numpy.float32)/9
box31 = cv2.filter2D(img1, -1, kernel3)
box32 = cv2.filter2D(img2, -1, kernel3)

kernel5 = numpy.ones((5,5), numpy.float32)/25
box51 = cv2.filter2D(img1, -1, kernel5)
box52 = cv2.filter2D(img2, -1, kernel5)

pyplot.subplot(231), pyplot.imshow(img1), pyplot.title('Original')
pyplot.xticks([]), pyplot.yticks([])
pyplot.subplot(232), pyplot.imshow(box31), pyplot.title('3x3 Box Filtered')
pyplot.xticks([]), pyplot.yticks([])
pyplot.subplot(233), pyplot.imshow(box51), pyplot.title('5x5 Box Filtered')
pyplot.xticks([]), pyplot.yticks([])
pyplot.subplot(234), pyplot.imshow(img2)
pyplot.xticks([]), pyplot.yticks([])
pyplot.subplot(235), pyplot.imshow(box32)
pyplot.xticks([]), pyplot.yticks([])
pyplot.subplot(236), pyplot.imshow(box52)
pyplot.xticks([]), pyplot.yticks([])
pyplot.show()