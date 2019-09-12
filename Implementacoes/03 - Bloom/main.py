import cv2 as cv
import numpy as np
import copy as cp
from bloomFunc import *

# parameters for bloom using box filter
NF = 5 # number of times to pass box filter to get gaussian
BOX_SIZE = 19

# parameters for bloom using gaussian
SIGMA = 15

# parameters for bloom
N_FILTERS = 5 # number of times to filter "bright" image
T = 0.5
ALPHA = 1
BETA = 0.1


IMG_IN = "Wind Waker GC.bmp"
IMG_OUT_BOX = "Wind Waker GC BOX.bmp"
IMG_OUT_GAUSSIAN = "Wind Waker GC GAUSSIAN.bmp"

img = cv.imread(IMG_IN, cv.IMREAD_COLOR)
img = cv.normalize(img.astype('float'), None, 0.0, 1.0, cv.NORM_MINMAX)

filterBox = [np.zeros(img.shape) for i in range(0, N_FILTERS)]
filterGaussian = [np.zeros(img.shape) for i in range(0, N_FILTERS)]

filtered = brightPass(img, T)

# filter images
filterBox[0] = cv.boxFilter(filtered, -1, (BOX_SIZE, BOX_SIZE))
for i in range(1, NF):
    filterBox[0] = cv.boxFilter(filterBox[0], -1, (BOX_SIZE, BOX_SIZE))
filterGaussian[0] = cv.GaussianBlur(filtered, (0, 0), SIGMA)

for i in range(1, N_FILTERS):
    filterBox[i] = cv.boxFilter(filtered, -1, (BOX_SIZE, BOX_SIZE))
    for j in range(1, (1+i)*NF):
        filterBox[i] = cv.boxFilter(filterBox[i], -1, (BOX_SIZE, BOX_SIZE))
    filterGaussian[i] = cv.GaussianBlur(filtered, (0, 0), (i+1)*SIGMA)

# sum filtered images
boxAdd = np.zeros(img.shape)
gaussianAdd = np.zeros(img.shape)
for i in range(0, N_FILTERS):
    boxAdd = np.add(boxAdd, filterBox[i])
    gaussianAdd = np.add(boxAdd, filterGaussian[i])

# final image g = img*alpha + sumFilters*beta
imgBox = ALPHA*img + BETA*boxAdd
imgGaussian = ALPHA*img + BETA*gaussianAdd

aux = np.zeros(img.shape, dtype=np.ubyte)
aux = imgBox*255
imgBox = cp.copy(aux)

aux = imgGaussian*255
imgGaussian = cp.copy(aux)

# saving images
cv.imwrite(IMG_OUT_BOX, imgBox)
cv.imwrite(IMG_OUT_GAUSSIAN, imgGaussian)
