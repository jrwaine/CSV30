import cv2 as cv
import numpy as np
from blob import *

IMG_IN = "documento-3mp.bmp"
IMG_OUT = "documento-3mp-Rotulado.bmp" 

NEGATIVO = True
THRESHOLD = 200
ALTURA_MIN = 2
LARGURA_MIN = 2
N_PIXELS_MIN = 4

BRANCO = 255
PRETO = 0
PIXEL_ROTULADO = 254 # deve ser diferente de "BRANCO" e "PRETO"

# binariza a imagem
def binariza(img):
    for y in range(len(img)):
        for x in range(len(img[y])):
            if(not NEGATIVO):
                if img[y, x] < THRESHOLD:
                    img[y, x] = PRETO
                else:
                    img[y, x] = BRANCO
            else:
                if img[y, x] > THRESHOLD:
                    img[y, x] = PRETO
                else:
                    img[y, x] = BRANCO
            

# funcao recursiva para rotular pixels e blob
def rotula(img, blob, y0, x0):
    img[y0, x0] = PIXEL_ROTULADO
    blob.addPixel(y0, x0)

    # vizinhança de 8
    for y in range(y0-1, y0+2):
        if(y < len(img) and y >= 0):
            for x in range(x0-1, x0+2):
                if (x >= 0 and x < len(img[y])):
                    if(img[y, x] == BRANCO):
                        rotula(img, blob, y, x)

# funcao para validar o blob
def validaBlob(blob):
    if (blob.nPixel < N_PIXELS_MIN):
        return False
    if((blob.xmax - blob.xmin) < LARGURA_MIN):
        return False
    if((blob.ymax - blob.ymin) < ALTURA_MIN):
        return False
    return True

img = cv.imread(IMG_IN, cv.IMREAD_GRAYSCALE)

binariza(img)
blobs = []

# rotulando
for y in range(0, len(img)):
    for x in range(0, len(img[y])):
        if(img[y, x] == BRANCO):
            blobs.append(Blob())
            rotula(img, blobs[-1], y, x)

blobsValidos = []

# valida blobs
for blob in blobs:
    if(validaBlob(blob)):
        blobsValidos.append(blob)

# desenha retangulos em blobs
img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
for blob in blobsValidos:
    cv.rectangle(img, (blob.xmin, blob.ymin), (blob.xmax, blob.ymax), (0,0,255), 1)

cv.namedWindow(IMG_OUT, cv.WINDOW_NORMAL)
cv.imshow(IMG_OUT, img)
cv.imwrite(IMG_OUT, img)
cv.waitKey(0)
cv.destroyAllWindows()

