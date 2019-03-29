import cv2
import numpy
import matplotlib
from matplotlib import pyplot

# Suppress runtimewarning for dividing by zero
numpy.seterr(divide='ignore', invalid='ignore')

# Input: Two sequenced images
# Prints the second image with colored velocity vectors at key points
def lucasKanade(img1, img2):
    # Parameters for GaussianBlur
    gauss_params = dict( ksize = (5, 5),
                         sigmaX = 1 )
    
    # Parameters for goodFeaturesToTrack
    maxCorners = 250
    feat_params = dict( maxCorners = maxCorners,
                        qualityLevel = .008,
                        minDistance = 10,
                        blockSize = 10)

    # Calculate I_t and apply a Gaussian Blur
    iT = img2 - img1
    iT = cv2.GaussianBlur(iT, **gauss_params)

    # Calculate I_x and I_y and apply a Gaussian Blur
    x1, y1 = numpy.gradient(img1)
    x2, y2 = numpy.gradient(img2)
    iX = (x1 + x2) // 2
    iY = (y1 + y2) // 2
    iX = cv2.GaussianBlur(iX, **gauss_params)
    iY = cv2.GaussianBlur(iY, **gauss_params)

    # Calculate the tensor representations in the LK formula
    kernel = numpy.ones((5, 5)) / 25
    tXX = cv2.filter2D(iX * iX, -1, kernel)
    tYY = cv2.filter2D(iY * iY, -1, kernel)
    tXY = cv2.filter2D(iX * iY, -1, kernel)
    tXT = cv2.filter2D(iX * iT, -1, kernel)
    tYT = cv2.filter2D(iY * iT, -1, kernel)
    
    # Calculate the u and v values for change in x and y positions
    u = ((tYT * tXY) - (tXT * tYY)) / ((tXX * tYY) - (tXY ** 2))
    v = ((tXT * tXY) - (tYT * tXX)) / ((tXX * tYY) - (tXY ** 2))

    # Find corners in the second image
    corners2 = cv2.goodFeaturesToTrack(img2, **feat_params)

    # Draw dots at all the found-corners (will be the head of the vectors)
    corner_pts2 = list()
    for i in range(len(corners2)):
        x, y = corners2[i].ravel()
        corner_pts2.append((int(x), int(y)))
        cv2.circle(img2, (x, y), 1, 255, -1)

    # Plot the second image
    pyplot.subplot(111), pyplot.imshow(img2, cmap='gray'), pyplot.title('Lucas-Kanade')
    pyplot.xticks([]), pyplot.yticks([])

    # Draw randomly colored lines for double the velocity at each corner found before
    hsv = matplotlib.cm.get_cmap('hsv')
    height = img2.shape[0]
    length = len(corner_pts2)
    for i in range(length):
        x, y = corner_pts2[i]
        color = hsv(int(i/length * 255))
        if x < height:
            pyplot.arrow(x, y, 2 * u[x][y], 2 * v[x][y], color=color)
        else:
            pyplot.arrow(x, y, 2 * u[y][x], 2 * v[y][x], color=color)
    
    # Show the image
    pyplot.show()

# Open the images
img1 = cv2.imread("./basketball1.png", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("./basketball2.png", cv2.IMREAD_GRAYSCALE)

# Apply the Lucas-Kanade algorithm to the images
lucasKanade(img1, img2)