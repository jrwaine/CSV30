import cv2 as cv
import numpy as np
import copy as cp
import param as pm
import funcs as fc

IMGS_IN = [ \
    "60.bmp",
    "82.bmp",
    "114.bmp",
    "150.bmp",
    "205.bmp"]

IMGS_OUT = [ \
    "60_out.bmp",
    "82_out.bmp",
    "114_out.bmp",
    "150_out.bmp",
    "205_out.bmp"]

LOG_PATH = 'log.txt'


with open(LOG_PATH, 'w') as f:
    f.write("PARAMETERS:\n")
    f.write("Box size: %d\n" % pm.BOX_SIZE)
    f.write("Treshold: %.2f\n" % pm.LIM)
    f.write("Min val. pixel: %.2f\n" % pm.MIN_L)
    f.write("Kernel size: %d\n" % pm.KERNEL_SIZE)
    f.write("\n")
    f.write("IMAGES:\n")

for i in range(0, len(IMGS_IN)):
    img = cv.imread(IMGS_IN[i], cv.IMREAD_COLOR) # read image
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY) # convert to gray scale
    # normalize to [0,1]
    img = cv.normalize(img.astype('float'), None, 0.0, 1.0, cv.NORM_MINMAX)
    # average image
    imgAvg = cv.blur(img, (pm.BOX_SIZE, pm.BOX_SIZE))
    # binary image 
    imgBin = np.where(np.logical_and(img >= imgAvg + pm.LIM, img > pm.MIN_L), 1.0, 0.0)
    # img to classify rice (erode image)
    kernel = np.ones((pm.KERNEL_SIZE, pm.KERNEL_SIZE), np.uint8)
    #imgCls = cv.erode(imgBin, kernel, iterations=1)
    imgCls = imgBin    
    # get rice blobs
    blobs = fc.getBlobs(imgCls)
    blobs = fc.validaBlobs(blobs)
    # image to save
    img = cv.cvtColor((imgCls*255).astype('uint8'), cv.COLOR_GRAY2RGB)

    # draw red rectangles on blobs
    for blob in blobs:
        cv.rectangle(img, (blob.xmin, blob.ymin), (blob.xmax, blob.ymax), (0,0,255), 1)
    '''
    cv.imshow('imgBin', imgBin)
    cv.waitKey()
    cv.imshow('imgCls', imgCls)
    cv.waitKey()
    cv.imshow('imgRice', img)
    cv.waitKey()
    '''
    # save image
    cv.imwrite(IMGS_OUT[i], img)
    
    # write log
    with open(LOG_PATH, 'a') as f:
        f.write(IMGS_IN[i] + "\n")
        f.write("    number of rices: %d\n" % len(blobs))
cv.destroyAllWindows()
