from img_filters import *

WIDTH = 5
HEIGHT = 5

IMG_IN = "documento-3mp.bmp"
IMG_OUT = "documento-3mp-5x5-integral.bmp"

imgAux = cv.imread(IMG_IN, cv.IMREAD_GRAYSCALE)
img = cv.normalize(imgAux.astype('float'), None, 0.0, 1.0, cv.NORM_MINMAX)

# img = filter_avg_separated(img, HEIGHT, WIDTH)
# img = filter_avg_naive(img, HEIGHT, WIDTH)
img = filter_avg_integral_img(img, HEIGHT, WIDTH)


'''
for y in range(HEIGHT, img.shape[0]-HEIGHT):
    for x in range(WIDTH, img.shape[1]-WIDTH):
        diff1 = img1[y,x]-img2[y,x]
        diff2 = img1[y,x]-img3[y,x]
        diff3 = img3[y,x]-img2[y,x]
        print("1 2 3", diff1, diff2, diff3)
'''
# img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
aux = np.zeros(img.shape, dtype = np.ubyte)
aux = img*255
img = aux

cv.imwrite(IMG_OUT, img)
# cv.namedWindow(IMG_OUT, cv.WINDOW_NORMAL)
# cv.imshow(IMG_OUT, img)
# cv.waitKey(0)
# cv.destroyAllWindows()