import numpy as np

# functions for bloom effect
def brightPass(img, T):
    imgAux = np.zeros(img.shape)
    for y in range(0, img.shape[0]):
        for x in range(0, img.shape[1]):
            if np.average(img[y, x]) >= T:
                imgAux[y, x, 0] = img[y, x, 0]
                imgAux[y, x, 1] = img[y, x, 1]
                imgAux[y, x, 2] = img[y, x, 2]
    return imgAux