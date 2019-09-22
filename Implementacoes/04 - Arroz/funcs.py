from blob import *

WHITE = 1.0
BLACK = 0.0
ROTULATED = 0.99 # value of rotulated pixel

# funcao recursiva para rotular pixels e blob
def rotula(img, blob, y0, x0):
    img[y0, x0] = ROTULATED
    blob.addPixel(y0, x0)

    # vizinhança de 8
    for y in range(y0-1, y0+2):
        if(y < len(img) and y >= 0):
            for x in range(x0-1, x0+2):
                if (x >= 0 and x < len(img[y])):
                    if(img[y, x] == WHITE):
                        rotula(img, blob, y, x)

# get list of blobs in image
def getBlobs(img):
    blobs = []
    for y in range(0, len(img)):
        for x in range(0, len(img[y])):
            if(img[y, x] == WHITE):
                blobs.append(Blob())
                rotula(img, blobs[-1], y, x)
    return blobs