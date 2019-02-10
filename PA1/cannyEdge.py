# cannyEdge.py
# John Hacker's Canny Edge Detection
# CAP 4453 Spring 19

'''
    I showed all the intermediate steps of sigma=1 and then additionally
showed the final result for sigma=2 and sigma=3. My canny filter worked 
best when sigma=1. I believe this is the case because it made the image 
less blurry than higher sigmas but still smoothed the image.
    The best image, threshold, sigma combination I found in all my testing
with my Canny filter is image1.jpg with low threshold of 60 and high
threshold of 90 and a sigma=1.
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

# Input: image, kenerl size, sigma value
# Output: image with gaussian filter applied
def fastGauss(img, sigma):
    # Convert image to array
    imgArr = numpy.array(img, dtype=float)

    # Find the size of the kernel
    size = sigma * 6

    # Make sure the kernel side is odd sized
    if (size % 2 == 0):
        size = size + 1

    # Create the kernel
    kernel = numpy.ones((1,size))

    # Find the offset from the center of the kernel
    offset = (size // 2) * -1

    # Loop through all the pixles in the kernel
    for i in range(0, size):
        # Find x and y for the gaussian function
        x = i + offset

        # Calculate the gaussian value for the current kernel pixel
        kernel[0][i] = (1 / (numpy.sqrt(2 * numpy.pi) * sigma)) * numpy.exp(-1 * (numpy.square(x)) / (2 * numpy.square(sigma)))

    # Return the smoothed image
    return gaussY(cv2.filter2D(imgArr, -1, kernel), sigma)

# Input: image, kenerl size, sigma value
# Output: image with gaussian filter applied
def gaussY(img, sigma):
    # Find the size of the kernel
    size = sigma * 6

    # Make sure the kernel side is odd sized
    if (size % 2 == 0):
        size = size + 1
    
    # Create the kernel
    kernel = numpy.ones((size, 1), numpy.float32)

    # Find the offset from the center of the kernel
    offset = (size // 2) * -1

    # Loop through all the pixles in the kernel
    for i in range(0, size):
        # Find x and y for the gaussian function
        y = i + offset

        # Calculate the gaussian value for the current kernel pixel
        kernel[i][0] = (1 / (numpy.sqrt(2 * numpy.pi) * sigma)) * numpy.exp(-1 * (numpy.square(y)) / (2 * numpy.square(sigma)))

    # Return the smoothed image
    return cv2.filter2D(img, -1, kernel)


# Input: image
# Output: image with central finite differene applied
def centralX(img):
    # Create the kernel
    kernel = numpy.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])

    # Return the smoothed image
    return cv2.filter2D(img, -1, kernel)

# Input: image
# Output: image with central finite differene applied
def centralY(img):
    # Create the kernel
    kernel = numpy.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    # Return the smoothed image
    return cv2.filter2D(img, -1, kernel)

# Returns an image that is the magnitde of two given images
def gradientMag(imgX, imgY):
    # Find the size of the image
    height, width = imgX.shape

    #imgMag = numpy.ones((width, height, 3), numpy.float32)
    imgMag = numpy.ndarray((height, width))

    # Loop through all the pixels in the kernel
    for i in range(0, height):
        for j in range(0, width):
            # Make the current pixel the magnitude of the other two images
            imgMag[i][j] = int(numpy.sqrt((numpy.square(imgX[i][j])) + (numpy.square(imgY[i][j]))))

    return imgMag

# Returns an array of all the orientatations between two images
def gradientOrient(imgX, imgY):
    # Find the angles between the y and x gradients
    directions = numpy.arctan(imgY, imgX) * (180 / numpy.pi)

    # Find the shape of the images
    height, width = directions.shape
    
    # Loop through all the pixels in the images
    for i in range(height):
        for j in range(width):
            # Check if angle is negative
            if directions[i][j] < 0:
                # Reflect negative angles
                directions[i][j] += 180

            # Round to 45
            if (22.5 <= directions[i][j] < 67.5):
                directions[i][j] = 45
            # Round to 90
            elif (67.5 <= directions[i][j] < 112.5):
                directions[i][j] = 90
            # Round to 135
            elif (112.5 <= directions[i][j] < 157.5):
                directions[i][j] = 135
            # Round to 0
            else:
                directions[i][j] = 0

    return directions


# Returns an image with edges that are 1-pixel wide
def suppress(mag, directions):
    #Find the shape of the images
    height, width = directions.shape
    
    # Loop through all the pxels in the arrays
    for i in range(height):
        for j in range(width):
            # 0 degrees
            if (directions[i][j] == 0):
                # Left column
                if (j == 0):
                    if (mag[i][j] <= mag[i][j + 1]):
                        mag[i][j] = 0
                # Right column
                elif (j == width - 1):
                    if (mag[i][j] <= mag[i][j - 1]):
                        mag[i][j] = 0
                # Somewhere in between
                else:
                    if (mag[i][j] <= mag[i][j + 1] or mag[i][j] <= mag[i][j - 1]):
                        mag[i][j] = 0
            # 90 degrees
            elif (directions[i][j] == 90):
                # Top row
                if (i == 0):
                    if (mag[i][j] <= mag[i + 1][j]):
                        mag[i][j] = 0
                # Bottom row
                elif (i == height - 1):
                    if (mag[i][j] <= mag[i - 1][j]):
                        mag[i][j] = 0
                # Somewhere in between
                else:
                    if (mag[i][j] <= mag[i + 1][j] or mag[i][j] <= mag[i - 1][j]):
                        mag[i][j] = 0
            # 45 degrees
            elif (directions[i][j] == 45):
                # Top-left corner or bottom-right corner
                if ((i == 0 and j == 0) or (i == height - 1 and j == width - 1)):
                    mag[i][j] = 0
                # Bottom row or right column
                elif (i == height - 1 or j == 0):
                    if (mag[i][j] <= mag[i - 1][j + 1]):
                        mag[i][j] = 0
                # Top row or left column
                elif (i == 0 or j == width - 1):
                    if (mag[i][j] <= mag[i + 1][j - 1]):
                        mag[i][j] = 0
                # Somewhere in between
                else:
                    if (mag[i][j] <= mag[i - 1][j + 1] or mag[i][j] <= mag[i + 1][j - 1]):
                        mag[i][j] = 0
            # 135 degrees
            else:
                # Bottom-left corner or top-right corner
                if ((i == height - 1 and j == 0) or (i == 0 and j == width - 1)):
                    mag[i][j] = 0
                # Bottom row or right column
                elif (i == height - 1 or j == width - 1):
                    if (mag[i][j] <= mag[i - 1][j - 1]):
                        mag[i][j] = 0
                # Top row or bottom column
                elif (i == 0 or j == 0):
                    if (mag[i][j] <= mag[i + 1][j + 1]):
                        mag[i][j] = 0
                # Somewhere in between
                else:
                    if (mag[i][j] <= mag[i - 1][j - 1] or mag[i][j] <= mag[i + 1][j + 1]):
                        mag[i][j] = 0
    
    return mag


# Returns an image with only strong or weak connected to strong edges
def hysteresis(img, lowT):
    # Find the shape of the image
    height, width = img.shape

    # Make an array to use for comparisons
    strengths = numpy.ndarray((height, width))

    # Edge strengths
    strong = 1
    weak = 0

    # Calculate high threshold based on the low threshold
    highT = 1.5 * lowT

    # Loop through all the pixels in the image
    for i in range(height):
        for j in range(width):
            # Strong edge
            if (img[i][j] > highT):
                strengths[i][j] = strong
            # Weak edge
            elif (lowT <= img[i][j] <= highT):
                strengths[i][j] = weak
            # No edge
            else:
                img[i][j] = 0

    # Loop through all the pixels in the image
    for i in range(height):
        for j in range(width):
            # Make decisions on weak pixels
            if (strengths[i][j] == weak):
                # Top row
                if (i == 0):
                    # Left column
                    if (j == 0):
                        # Check for strong edge connection
                        if not (strengths[i + 1][j] == strong or strengths[i][j + 1] == strong or strengths[i + 1][j + 1] == strong):
                            img[i][j] = 0
                    # Right column
                    elif (j == width - 1):
                        # Check for strong edge connection
                        if not (strengths[i + 1][j] == strong or strengths[i][j - 1] == strong or strengths[i + 1][j - 1] == strong):
                            img[i][j] = 0
                    # Somewhere in between
                    else:
                        # Check for strong edge connection
                        if not (strengths[i + 1][j] == strong or strengths[i][j + 1] == strong or strengths[i][j - 1] == strong or strengths[i + 1][j + 1] or strengths[i + 1][j - 1] == strong):
                            img[i][j] = 0
                # Bottom row
                elif (i == height - 1):
                    # Left column
                    if (j == 0):
                        # Check for strong edge connection
                        if not (strengths[i - 1][j] == strong or strengths[i][j + 1] == strong or strengths[i - 1][j + 1] == strong):
                            img[i][j] = 0
                    # Right column
                    elif (j == width - 1):
                        # Check for strong edge connection
                        if not (strengths[i - 1][j] == strong or strengths[i][j - 1] == strong or strengths[i - 1][j - 1] == strong):
                            img[i][j] = 0
                    # Somewhere in between
                    else:
                        # Check for strong edge connection
                        if not (strengths[i - 1][j] == strong or strengths[i][j + 1] == strong or strengths[i][j - 1] == strong or strengths[i - 1][j + 1] or strengths[i - 1][j - 1] == strong):
                            img[i][j] = 0
                # Somewhere in between
                else:
                    # Left column
                    if (j == 0):
                        # Check for strong edge connection
                        if not (strengths[i + 1][j] == strong or strengths[i - 1][j] == strong or strengths[i][j + 1] == strong or strengths[i + 1][j + 1] or strengths[i - 1][j + 1] == strong):
                            img[i][j] = 0
                    # Right column
                    elif (j == width - 1):
                        # Check for strong edge connection
                        if not (strengths[i + 1][j] == strong or strengths[i][j - 1] == strong or strengths[i + 1][j - 1] or strengths[i - 1][j - 1] == strong):
                            img[i][j] = 0
                    # Somewhere in between
                    else:
                        # Check for strong edge connection
                        if not (strengths[i + 1][j] == strong or strengths[i - 1][j] == strong or strengths[i][j + 1] == strong or strengths[i][j - 1] == strong or strengths[i + 1][j + 1] or strengths[i + 1][j - 1] or strengths[i - 1][j + 1] or strengths[i - 1][j - 1] == strong):
                            img[i][j] = 0
    
    return img

# Prints images with all the steps of applying the canny filter to an image
def canny(img, lowT, sig1, sig2, sig3):
    # Convert the image to gray scale
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply gauss filter
    gauss = fastGauss(imgGray, sig1)

    # Apply gradient operations (x-central, y-central, magnitude, and orientation)
    gradX = centralX(gauss)
    pyplot.subplot(333), pyplot.imshow(gradX, cmap='gray'), pyplot.title('Gradient X')
    pyplot.xticks([]), pyplot.yticks([])
    gradY = centralY(gauss)
    gradMag = gradientMag(gradX, gradY)
    pyplot.subplot(335), pyplot.imshow(gradMag, cmap='gray'), pyplot.title('Gradient Magnitude')
    pyplot.xticks([]), pyplot.yticks([])
    gradDir = gradientOrient(gradX, gradY)
    gradDir = gradDir.astype(int)
    
    # Apply non-max suppression
    sup = suppress(gradMag, gradDir)
    pyplot.subplot(337), pyplot.imshow(sup, cmap='gray'), pyplot.title('Non-Max Suppressed')
    pyplot.xticks([]), pyplot.yticks([])

    # Apply hysteresis thresholding
    hyst = hysteresis(sup, lowT)

    gauss = gauss.astype(int)

    # Plot the images
    pyplot.subplot(331), pyplot.imshow(img, cmap='gray'), pyplot.title('Original')
    pyplot.xticks([]), pyplot.yticks([])
    pyplot.subplot(332), pyplot.imshow(gauss, cmap='gray'), pyplot.title('Gauss')
    pyplot.xticks([]), pyplot.yticks([])
    pyplot.subplot(334), pyplot.imshow(gradY, cmap='gray'), pyplot.title('Gradient Y')
    pyplot.xticks([]), pyplot.yticks([])
    pyplot.subplot(336), pyplot.imshow(gradDir, cmap='gray'), pyplot.title('Gradient Orientation')
    pyplot.xticks([]), pyplot.yticks([])
    pyplot.subplot(338), pyplot.imshow(hyst, cmap='gray'), pyplot.title('Canny Filtered')
    pyplot.xticks([]), pyplot.yticks([])
    pyplot.show()

    # Repeat process with different sigma values
    gaussSig1 = fastGauss(imgGray, sig2)
    gaussSig2 = fastGauss(imgGray, sig3)
    gradXSig1 = centralX(gaussSig1)
    gradXSig2 = centralX(gaussSig2)
    gradYSig1 = centralY(gaussSig1)
    gradYSig2 = centralY(gaussSig2)
    gradMagSig1 = gradientMag(gradXSig1, gradYSig1)
    gradMagSig2 = gradientMag(gradXSig2, gradYSig2)
    gradDirSig1 = gradientOrient(gradXSig1, gradYSig1)
    gradDirSig2 = gradientOrient(gradXSig2, gradYSig2)
    supSig1 = suppress(gradMagSig1, gradDirSig1)
    supSig2 = suppress(gradMagSig2, gradDirSig2)
    hystSig1 = hysteresis(supSig1, lowT)
    hystSig2 = hysteresis(supSig2, lowT)

    # Plot orginal, and both completes canny filters with different sigmas
    pyplot.subplot(221), pyplot.imshow(img, cmap='gray'), pyplot.title('Original')
    pyplot.xticks([]), pyplot.yticks([])
    pyplot.subplot(222), pyplot.imshow(hyst, cmap='gray'), pyplot.title('Canny sigma=' + str(sig1))
    pyplot.xticks([]), pyplot.yticks([])
    pyplot.subplot(223), pyplot.imshow(hystSig1, cmap='gray'), pyplot.title('Canny sigma=' + str(sig2))
    pyplot.xticks([]), pyplot.yticks([])
    pyplot.subplot(224), pyplot.imshow(hystSig2, cmap='gray'), pyplot.title('Canny sigma=' + str(sig3))
    pyplot.xticks([]), pyplot.yticks([])
    pyplot.show()

# Open both images
img1 = cv2.imread("./Images/canny1.jpg")
img2 = cv2.imread("./Images/canny2.jpg")

# Apply canny filter to both images
canny(img1, 60, 1, 2,  3)
canny(img2, 60, 1, 2, 3)