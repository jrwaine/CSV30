import cv2 as cv
import numpy as np
import copy as cp
import param as pm

IMGS_IN = [ \
    "img/0.bmp",
    "img/1.bmp",
    "img/2.bmp",
    "img/3.bmp",
    "img/4.bmp",
    "img/5.bmp",
    "img/6.bmp",
    "img/7.bmp",
    "img/8.bmp"]

IMGS_OUT = [ \
    "resultados/0_out.bmp",
    "resultados/1_out.bmp",
    "resultados/2_out.bmp",
    "resultados/3_out.bmp",
    "resultados/4_out.bmp",
    "resultados/5_out.bmp",
    "resultados/6_out.bmp",
    "resultados/7_out.bmp",
    "resultados/8_out.bmp"]

IMG_MASK = "img/Wind Waker GC.bmp"

for name in range(0, len(IMGS_IN)):

    img = cv.imread(IMGS_IN[name], cv.IMREAD_COLOR) # read image

    # increase image saturation for "bright" pixels
    imgHLS = cv.cvtColor(img, cv.COLOR_RGB2HLS)
    sat = imgHLS[:,:,2]
    sat = np.where(sat > pm.MIN_SAT, 255, sat)
    imgHLS[:,:,2] = sat
    imgHLS = cv.cvtColor(imgHLS, cv.COLOR_HLS2RGB)
    imgHLS = cv.cvtColor(img, cv.COLOR_RGB2YCrCb).astype('float32')
    
    # normalize values to (0, 1)
    imgHLS /= 255
    # gray image, distance to green in YCrCb
    imgGray = np.zeros(img.shape[:-1]).astype('float32')
    imgGray = ((imgHLS[:, :, 1]**2 + imgHLS[:, :, 2]**2))**0.5
    imgGray /= 2**0.5
    imgGray *= 255

    # normalize image, discarding NORM_PERC pixels
    histG = cv.calcHist([imgGray],[0],None,[256],[0,255])
    cont = 0
    nPixels = imgGray.shape[0]*imgGray.shape[1]
    minNorm, maxNorm = 0, 1
    for i in range(0, 256):
        cont += histG[i][0]
        if(cont > nPixels*pm.NORM_PERC/100):
            imgGray = np.where(imgGray < minNorm, i, imgGray)
        elif cont > nPixels*(100-pm.NORM_PERC)/100:
            imgGray = np.where(imgGray > maxNorm, i, imgGray)
            break
    
    cv.normalize(imgGray, imgGray, 0, 255, cv.NORM_MINMAX)

    # Treshold using Otsu for gray image
    trVal, imgBin = cv.threshold(imgGray.astype('uint8'),0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

    # histogram for "upper" and "lower" pixels
    hist1, hist2 = list(), list()
    for y in range(0, imgBin.shape[0]):
        for x in range(0, imgBin.shape[1]):
            if imgBin[y, x] <= trVal:
                hist1.append(imgGray[y, x])
            else:
                hist2.append(imgGray[y, x])
    hist1, hist2 = np.array(hist1), np.array(hist2)

    # limits to consider the pixel as unknow, using std deviation
    std1, std2 = np.std(hist1), np.std(hist2)
    lim1, lim2 = std1*pm.STD_DEV_MULTP+np.average(hist1), -std2*pm.STD_DEV_MULTP+np.average(hist2)
    print(std1, std2, lim1, lim2, np.max(hist1), np.min(hist2))
    # update unkown pixels
    imgBin = np.where(np.logical_or(np.logical_and(imgGray < trVal, imgGray > lim1), \
        np.logical_and(imgGray >= trVal, imgGray < lim2)), 128, imgBin)


    imgSave = cv.imread(IMG_MASK, cv.IMREAD_COLOR).astype('float32') # read mask image

    imgSave = cv.resize(imgSave, (img.shape[1], img.shape[0]))
    for y in range(0, len(imgBin)):
        for x in range(0, len(imgBin[y])):
            if(imgBin[y, x] == 255):
                imgSave[y, x] = img[y, x]
            elif imgBin[y, x] == 128:
                imgSave[y, x] = img[y, x]//2+imgSave[y, x]//2

    # image to save
    
    # save image
    cv.imwrite(IMGS_OUT[name], imgSave.astype('uint8'))
    
cv.destroyAllWindows()