import cv2 as cv
import numpy as np
import copy as cp
import param as pm
import time
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


def getDist(imgHLS):
    imgGray = np.zeros(img.shape[:-1]).astype('float32')
    imgGray = ((imgHLS[:, :, 1]**2 + imgHLS[:, :, 2]**2))**0.5
    imgGray /= 2**0.5
    return imgGray


def getDistRealGreen(imgHLS, realGreen):
    imgGray = np.zeros(img.shape[:-1]).astype('float32')
    imgGray = ((imgHLS[:, :, 1]-realGreen[1])**2 + (imgHLS[:, :, 2]-realGreen[2])**2)**0.5
    imgGray /= 3**0.5
    return imgGray

def lowerGreenSaturation(img):
    imgHLS = cv.cvtColor(img, cv.COLOR_RGB2HLS)
    sat = imgHLS[:,:,2]
    hue = imgHLS[:,:,0]
    sat = np.where(np.logical_and(hue > (90-22.5)/2, hue < (135+22.5)/2), 20, sat)
    imgHLS[:,:,2] = sat
    img = cv.cvtColor(imgHLS, cv.COLOR_HLS2RGB)
    return img

def getRealGreen(imgBin, imgHLS):
    realGreen = np.array([0.0, 0.0, 0.0])
    total = 0
    for y in range(0, len(imgBin)):
        for x in range(0, len(imgBin[y])):
            if(imgBin[y, x] == 0):
                realGreen[0] += imgHLS[y, x, 0]
                realGreen[1] += imgHLS[y, x, 1]
                realGreen[2] += imgHLS[y, x, 2]
                total += 1
    realGreen /= total
    print(realGreen, total)
    return realGreen


for name in range(0, len(IMGS_IN)):

    times = []
    img = cv.imread(IMGS_IN[name], cv.IMREAD_COLOR) # read image

    # increase image saturation for "bright" pixels
    imgHLS = cv.cvtColor(img, cv.COLOR_RGB2HLS)

    sat = imgHLS[:,:,2]
    sat = np.where(sat > pm.MIN_SAT, 200, sat)
    imgHLS[:,:,2] = sat

    imgHLS = cv.cvtColor(imgHLS, cv.COLOR_HLS2RGB)
    imgHLS = cv.cvtColor(img, cv.COLOR_RGB2YCrCb).astype('float32')
    
    # normalize values to (0, 1)
    imgHLS /= 255
    # gray image, distance to green in YCrCb
    t1 = time.time()
    imgGray = getDist(imgHLS)
    times.append(time.time()-t1) # 1
    imgGray *= 255

    cv.normalize(imgGray, imgGray, 0, 255, cv.NORM_MINMAX)

    # normalize image, discarding NORM_PERC pixels
    t1 = time.time()
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
    times.append(time.time()-t1) # 2
    
    # Treshold using Otsu for gray image
    trVal, imgBin = cv.threshold(imgGray.astype('uint8'),0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    '''
    t1 = time.time()
    realGreen = getRealGreen(imgBin, imgHLS)
    imgGray = getDistRealGreen(imgHLS, realGreen)
    times.append(time.time()-t1) # 3
    imgGray *= 255
    cv.normalize(imgGray, imgGray, 0, 255, cv.NORM_MINMAX)

    trVal, imgBin = cv.threshold(imgGray.astype('uint8'),0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    '''

    # histogram for "upper" and "lower" pixels
    t1 = time.time()
    hist1, hist2 = list(), list()
    for y in range(0, imgBin.shape[0]):
        for x in range(0, imgBin.shape[1]):
            if imgBin[y, x] <= trVal:
                hist1.append(imgGray[y, x])
            else:
                hist2.append(imgGray[y, x])
    hist1, hist2 = np.array(hist1), np.array(hist2)
    times.append(time.time()-t1) # 4
    lim1 = min(trVal, np.average(hist1)+30)
    lim2 = trVal+5
    # limits to consider the pixel as unknow, using std deviation
    #std1, std2 = np.std(hist1), np.std(hist2)
    
    #lim1, lim2 = std1*pm.STD_DEV_MULTP+np.average(hist1), -std2*pm.STD_DEV_MULTP+np.average(hist2)
    #print(std1, std2, lim1, lim2, np.max(hist1), np.min(hist2))
    
    # update unkown pixels
    imgBin = np.where((np.logical_and(imgGray < lim2, imgGray > lim1)), 128, imgBin)
    

    imgMask = cv.imread(IMG_MASK, cv.IMREAD_COLOR).astype('float32') # read mask image

    imgMask = cv.resize(imgMask, (img.shape[1], img.shape[0]))
    imgSave = np.zeros(img.shape)
    t1 = time.time()
    img = lowerGreenSaturation(img)
    times.append(time.time()-t1) # 5
    t1 = time.time()
    for y in range(0, len(imgBin)):
        for x in range(0, len(imgBin[y])):
            if(imgBin[y, x] == 255) or imgBin[y, x] == 128:
                imgSave[y, x] = img[y, x]
            else: 
                imgSave[y, x] = imgMask[y, x]
            '''
            elif imgBin[y, x] == 128:
                weightMask = 255-(imgGray[y,x]-trVal)
                weightImg = 255-(imgGray[y, x]-np.average(hist1))
                imgSave[y, x] = (imgMask[y, x] * weightMask + img[y, x] * weightImg)
                imgSave[y, x] /= (weightImg+weightMask)
            '''
    times.append(time.time()-t1) # 6
    print(times)
    # save image
    cv.imwrite(IMGS_OUT[name], imgSave.astype('uint8'))
    
cv.destroyAllWindows()