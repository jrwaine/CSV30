from blob import *
import cv2 as cv
import param as pm
import numpy as np

WHITE = 1.0
BLACK = 0.0
ROTULATED = 0.99 # value of rotulated pixel

ALTURA_MIN = 2
LARGURA_MIN = 2
N_PIXELS_MIN = 4

# funcao recursiva para rotular pixels e blob
def rotula(img, blob, y0, x0):
    img[y0, x0] = ROTULATED
    blob.addPixel(y0, x0)
    pixels = list()
    pixels.append((y0, x0))
    while(len(pixels) > 0):
        y0, x0 = pixels[0]        
        del pixels[0]    
        # vizinhança de 8
        for y in range(y0-1, y0+2):
            if(y < len(img) and y >= 0):
                for x in range(x0-1, x0+2):
                    if (x >= 0 and x < len(img[y])):
                        if(img[y, x] == WHITE):
                            img[y, x] = ROTULATED
                            blob.addPixel(y, x)
                            pixels.append((y, x)) 

# get list of blobs in image
def getBlobs(img):
    blobs = []
    for y in range(0, len(img)):
        for x in range(0, len(img[y])):
            if(img[y, x] == WHITE):
                blobs.append(Blob())
                rotula(img, blobs[-1], y, x)
    return blobs

# funcao para validar o blob
def validaBlobs(blobs):
    valBlobs = []
    for blob in blobs:
        if (blob.nPixel < N_PIXELS_MIN):
            continue
        if((blob.xmax - blob.xmin) < LARGURA_MIN):
            continue
        if((blob.ymax - blob.ymin) < ALTURA_MIN):
            continue
        valBlobs.append(blob)
    return valBlobs


def checkDiscontinuity(blob, imgBin):
    disc = [0, 0]
    for x in range(blob.xmin, blob.xmax+1):
        val = imgBin[blob.ymin, x]
        change = 0
        for y in range(blob.ymin+1, blob.ymax+1):
            if imgBin[y, x] != val:
                val = imgBin[y, x]
                change += 1
                if change > 2:
                    disc[0] += 1
                    break
    
    for y in range(blob.ymin, blob.ymax+1):
        val = imgBin[y, blob.xmin]
        change = 0
        for x in range(blob.xmin+1, blob.xmax+1):
            if imgBin[y, x] != val:
                val = imgBin[y, x]
                change += 1
                if change > 2:
                    disc[1] += 1
                    break
    return disc


def erodeBlob(blob, imgBin):
    kernel = np.ones((pm.KERNEL_SIZE, pm.KERNEL_SIZE), np.uint8)
    imgErode = cv.erode(imgBin, kernel, iterations=1)
    for y, x in blob.pixels:
        imgBin[y, x] = imgErode[y, x]


def updateBlob(blobs, blobUpdate, imgBin):
    for y, x in blobUpdate.pixels:
        if(imgBin[y, x] > BLACK):
            imgBin[y, x] = WHITE
    newBlobs = []

    for y in range(blobUpdate.ymin, blobUpdate.ymax+1):
        for x in range(blobUpdate.xmin, blobUpdate.xmax+1):
            if(imgBin[y, x] == WHITE):
                newBlobs.append(Blob())
                rotula(imgBin, newBlobs[-1], y, x)
    return newBlobs


def treatBlobs(blobs, imgBin):
    newBlobs = []
    treatedBlobs = []
    for blob in blobs:
        disc = checkDiscontinuity(blob, imgBin)
        if(sum(disc) >= 2):
            erodeBlob(blob, imgBin)
            resBlobs = updateBlob(blobs, blob, imgBin)
            for resBlob in resBlobs:
                newBlobs.append(resBlob)
            treatedBlobs.append(blob)

    for blob in treatedBlobs:
        blobs.remove(blob)
    
    for blob in newBlobs:
        blobs.append(blob)
    
    return (treatedBlobs, newBlobs)