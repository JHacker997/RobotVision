import cv2
import numpy
import matplotlib
import control
from matplotlib import pyplot

'''
    My original Lucas-Kanade application shows horizontal movement in the basketball that makes
with a visual comparison of the two images. I use key points to track and find about 120 corners
in the image to track the movement at. I print the vectors on the second image to show were the
features were calculated to have moved from to where they are now. The white dots on the image
should be considered as the head of the arrows. Since the ball does not have many corners, it did
not get a lot of points returned for it by goodFeaturesToTrack. I experimented a lot on different
feature finding parameters to get a decenet amount to show up on the ball and hands.
    I adapted my code to have parameters for whether it is using a pyramid, the level on the 
pyramid, and the total amount of levels. I apply the same method as before on each level and print 
them side by side. I use a heuristical approach to dynamically subplot the images based on the amount
of levels in the pyramid. I found that each level increase in the pyramid made my output look 
worse. The kernel size did not appear to affect my outputs that much. I setteled on showing the
window size and max levels that I did because I thought it showed the progression of pyramid levels
the best.
    I provided the images in this directory to show for security in case this is somhow unable to 
run on your machine.
'''

# Suppress runtimewarning for dividing by zero
numpy.seterr(divide='ignore', invalid='ignore')

# Input: Two sequenced images, boolean pyramid use, level index, and max levels
# Prints the second image with colored velocity vectors at key points
def lucasKanade(img1, img2, pyr_mode=False, idx=0, retval=0):
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

    # Check if plotting a pyramid image
    if pyr_mode:
        # Find appropriate sizes for rows and cols
        if retval < 3:
            rows = 1
            cols = retval + 1
        else:
            rows = 2
            if retval == 3:
                cols = 2
            else:
                cols = 3

        # Calculate the subplot number
        sub = rows * 100 + cols * 10 + idx + 1

        # Plot the pyramid image
        pyplot.subplot(sub), pyplot.imshow(img2, cmap='gray'), pyplot.title('Pyramid Level ' + str(idx))
    else:
        # Plot the second image
        pyplot.subplot(111), pyplot.imshow(img2, cmap='gray'), pyplot.title('Lucas-Kanade')

    # Take off the x and y ticks
    pyplot.xticks([]), pyplot.yticks([])

    # Draw randomly colored lines for double the velocity at each corner found before
    hsv = matplotlib.cm.get_cmap('hsv')
    height = img2.shape[0]
    length = len(corner_pts2)
    for i in range(length):
        x, y = corner_pts2[i]
        color = hsv(int(i/length * 255))
        if x < height:
            pyplot.arrow(x, y, u[x][y], v[x][y], color=color)
        else:
            pyplot.arrow(x, y, u[y][x], v[y][x], color=color)

# Open the images
img1 = cv2.imread("./basketball1.png", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("./basketball2.png", cv2.IMREAD_GRAYSCALE)

# Apply the Lucas-Kanade algorithm to the images
lucasKanade(img1, img2)

# Show the image
pyplot.show()

pyr_params = dict( winSize = (3, 3),
                maxLevel = 3 )

# Create a pyramid of the image
retval, pyr1 = cv2.buildOpticalFlowPyramid(img1, **pyr_params)
retval, pyr2 = cv2.buildOpticalFlowPyramid(img2, **pyr_params)

# Apply Lucas-Kanade to all of the pyramid levels
for i in range(retval + 1):
    idx = i * 2
    lucasKanade(pyr1[idx], pyr2[idx], True, i, retval)

# Show the images
pyplot.show()