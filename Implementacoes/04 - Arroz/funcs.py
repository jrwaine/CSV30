from blob import *

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
