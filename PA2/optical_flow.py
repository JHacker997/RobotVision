import cv2
import numpy
import matplotlib
from matplotlib import pyplot
from PIL import Image

numpy.seterr(divide='ignore', invalid='ignore')

def lucasKanade():
    maxCorners = 1000
    feat_params = dict(maxCorners = maxCorners,
                       qualityLevel = .03,
                       minDistance = 5,
                       blockSize = 15)

    img1 = cv2.imread("./basketball1.png", cv2.IMREAD_GRAYSCALE)
    img1 = cv2.GaussianBlur(img1,(5,5),1)
    img2 = cv2.imread("./basketball2.png", cv2.IMREAD_GRAYSCALE)
    img2 = cv2.GaussianBlur(img2,(5,5),1)

    tT = img2 - img1
    tY, tX = numpy.gradient(img2)
    kernel = numpy.ones((5, 5))
    tXX = cv2.filter2D(tX * tX, -1, kernel)
    tYY = cv2.filter2D(tY * tY, -1, kernel)
    tXY = cv2.filter2D(tX * tY, -1, kernel)
    tXT = cv2.filter2D(tX * tT, -1, kernel)
    tYT = cv2.filter2D(tY * tT, -1, kernel)
    u = ((tYT * tXY) - (tXT * tYY)) // ((tXX * tYY) - (tXY ** 2))
    v = ((tXT * tXY) - (tYT * tXX)) // ((tXX * tYY) - (tXY ** 2))

    height, width = img1.shape
    corners1 = cv2.goodFeaturesToTrack(img1, **feat_params)
    print("corners1: " + str(len(corners1)))

    # print(str(u))
    corner_pts1 = list()
    for i in range(maxCorners):
        try:
            if corners1[i][0][0] < height and corners1[i][0][1] < width:
                corner_pts1.append((int(corners1[i][0][0]), int(corners1[i][0][1])))
        except:
            pass
    print("corner_pts1: " + str(len(corner_pts1)))
    print(str(height) + " x " + str(width))

    height, width = img2.shape
    corners2 = cv2.goodFeaturesToTrack(img2, **feat_params)
    print("corners2: " + str(len(corners2)))
    corner_pts2 = list()
    for i in range(maxCorners):
        try:
            if corners2[i][0][0] < height and corners2[i][0][1] < width:
                corner_pts2.append((int(corners2[i][0][0]), int(corners2[i][0][1])))
        except:
            pass
    print("corner_pts2: " + str(len(corner_pts2)))
    print(str(height) + " x " + str(width))

    numCorners1 = len(corner_pts1)
    numCorners2 = len(corner_pts2)

    # for i in range(numCorners2):
    #     x, y = corner_pts2[i]
    #     # d = tXX[x][y] * tYY[x][y] - tXY[x][y] ** 2
    #     # if not d == 0:
    #     #     u = tYT[x][y] * tXY[x][y] - tXT[x][y] * tYY[x][y] // d
    #     #     u = int(u)
    #     #     v = tXT[x][y] * tXY[x][y] - tYT[x][y] * tXX[x][y] // d
    #     #     v = int(v)
    #     print(str(u[x][y]) + " " + str(v[x][y]))
    #     cv2.line(img2, (x, y), (int(u[x][y]), int(v[x][y])), 255)

    for i in range(maxCorners):
        try:
            #print(str(corner_pts1[i]))
            cv2.circle(img1, corner_pts1[i], 1, 255, -1)
        except:
            print("error on " + str(i))
            break
    for i in range(maxCorners):
        try:
            #print(str(corner_pts2[i]))
            cv2.circle(img2, corner_pts2[i], 1, 255, -1)
        except:
            print("error on " + str(i))
            break
    
    hsv = matplotlib.cm.get_cmap('hsv')
    width, height = img2.shape

    pyplot.subplot(121), pyplot.imshow(img1, cmap='gray'), pyplot.title('First Frame')
    pyplot.xticks([]), pyplot.yticks([])
    pyplot.subplot(122), pyplot.imshow(img2, cmap='gray'), pyplot.title('Second Frame')
    freq = 10
    # for i in range (height):
    #     for j in range(width):
    #         if i % freq == 1 and j % freq == 1:
    #             try:
    #                 color = hsv((j+1)/(i+1))
    #                 pyplot.arrow(i, j, int(u[i][j]), int(v[i][j]), color=color)
    #             except:
    #                 pass
    for i in range(numCorners2):
        x, y = corner_pts2[i]
        color = hsv(1/(i+1))
        pyplot.arrow(x, y, int(u[x][y]), int(v[x][y]), color=color)
    pyplot.xticks([]), pyplot.yticks([])
    pyplot.show()
    # pyplot.subplot(131), pyplot.imshow(tT, cmap='gray'), pyplot.title('T_t')
    # pyplot.xticks([]), pyplot.yticks([])
    # pyplot.subplot(132), pyplot.imshow(tX, cmap='gray'), pyplot.title('T_x')
    # pyplot.xticks([]), pyplot.yticks([])
    # pyplot.subplot(133), pyplot.imshow(tY, cmap='gray'), pyplot.title('T_y')
    # pyplot.xticks([]), pyplot.yticks([])
    # pyplot.show()
    # pyplot.subplot(231), pyplot.imshow(tXX, cmap='gray'), pyplot.title('T_xx')
    # pyplot.xticks([]), pyplot.yticks([])
    # pyplot.subplot(232), pyplot.imshow(tYY, cmap='gray'), pyplot.title('T_yy')
    # pyplot.xticks([]), pyplot.yticks([])
    # pyplot.subplot(233), pyplot.imshow(tXY, cmap='gray'), pyplot.title('T_xy')
    # pyplot.xticks([]), pyplot.yticks([])
    # pyplot.subplot(234), pyplot.imshow(tXT, cmap='gray'), pyplot.title('T_xt')
    # pyplot.xticks([]), pyplot.yticks([])
    # pyplot.subplot(235), pyplot.imshow(tYT, cmap='gray'), pyplot.title('T_yt')
    # pyplot.xticks([]), pyplot.yticks([])
    pyplot.show()

#LK()

lucasKanade()