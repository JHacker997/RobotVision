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
    I adapted my code to have parameters for whether it is using a pyramid and the total amount of 
levels. I apply the LK algorithm on each level and add the results to the u adn v matrices. I 
found that each level increase in the pyramid made my output look worse. The kernel size did not 
appear to affect my outputs that much. I setteled on showing the window size and max levels that I 
did because I thought it showed the progression of pyramid levels the best. The lines on the ball
get bigger which is good because it should be ht ebiggest movement between the frames and the point
of the pyramids to find the bigger motions. There is a problem with some of the key points getting 
much longer lines than they should based on some problem in the applied algorithm.
    I provided the images in this directory to show for security in case this is somhow unable to 
run on your machine.
'''

# Suppress runtimewarning for dividing by zero
numpy.seterr(divide='ignore', invalid='ignore')

# Input: Two sequenced images, boolean pyramid use, level index, and max levels
# Prints the second image with colored velocity vectors at key points
def lucasKanade(first, second, pyr_mode=False, maxLvl=1):
    # Parameters for GaussianBlur
    gauss_params = dict( ksize = (5, 5),
                         sigmaX = 1 )
    
    # Parameters for goodFeaturesToTrack
    feat_params = dict( maxCorners = 250,
                        qualityLevel = .008,
                        minDistance = 10,
                        blockSize = 10 )

    # Set the images for the 
    if not pyr_mode:
        img1 = first
        img2 = second

    # Loop though every level of the input pyramid
    for lvl in range(maxLvl):
        # Update the pyramid level
        if pyr_mode:
            img1 = first[lvl * 2]
            img2 = second[lvl * 2]

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
        if lvl == 0:
            u = ((tYT * tXY) - (tXT * tYY)) / ((tXX * tYY) - (tXY ** 2))
            v = ((tXT * tXY) - (tYT * tXX)) / ((tXX * tYY) - (tXY ** 2))
        else:
            u1 = ((tYT * tXY) - (tXT * tYY)) / ((tXX * tYY) - (tXY ** 2))
            v1 = ((tXT * tXY) - (tYT * tXX)) / ((tXX * tYY) - (tXY ** 2))
            for i in range(u1.shape[0]):
                for j in range(u1.shape[1]):
                    u[i * 2 ** lvl][j * 2 ** lvl] += u1[i][j]
                    v[i * 2 ** lvl][j * 2 ** lvl] += v1[i][j]
    
    # Reset the second image to show
    if pyr_mode:
        img2 = second[0]

    # Find corners in the second image
    corners2 = cv2.goodFeaturesToTrack(img2, **feat_params)

    # Draw dots at all the found-corners (will be the head of the vectors)
    corner_pts2 = list()
    for i in range(len(corners2)):
        x, y = corners2[i].ravel()
        corner_pts2.append((int(x), int(y)))
        cv2.circle(img2, (x, y), 1, 255, -1)

    # Plot the second image
    pyplot.subplot(111), pyplot.imshow(img2, cmap='gray')

    # Adjust the title based on if pyramids are used
    if not pyr_mode:
        pyplot.title('Lucas-Kanade')
    else:
        pyplot.title('Lucas-Kanade with Pyramids')

    # Take off the x and y ticks
    pyplot.xticks([]), pyplot.yticks([])

    # Draw randomly colored lines for the velocity at each corner found before
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

    # Show the images
    pyplot.show()

# Parameters for making a pyramid
kSize = 9
pyr_params = dict( winSize = (kSize, kSize),
                   maxLevel = 4 )

# Open the images
img1 = cv2.imread("./basketball1.png", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("./basketball2.png", cv2.IMREAD_GRAYSCALE)

# Create a pyramid of the image
retval, pyr1 = cv2.buildOpticalFlowPyramid(img1, **pyr_params)
retval, pyr2 = cv2.buildOpticalFlowPyramid(img2, **pyr_params)

# Apply the Lucas-Kanade algorithm to the images
lucasKanade(img1, img2)

# Apply the Lucas_kanade algorithm to the pyramids
lucasKanade(pyr1, pyr2, True, retval + 1)