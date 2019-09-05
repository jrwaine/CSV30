from img_filters import *

WIDTH = 3
HEIGHT = 3

IMG_IN = "flower.bmp"
IMG_OUT = "flower-3x3-separated.bmp"

imgAux = cv.imread(IMG_IN, cv.IMREAD_COLOR)
imgAux = cv.normalize(imgAux.astype('float'), None, 0.0, 1.0, cv.NORM_MINMAX)
img = np.zeros(np.shape(imgAux))

for i in range(0, np.shape(img)[-1]):
	img[:,:,i] = filter_avg_separated(imgAux[:,:,i], HEIGHT, WIDTH)
	# img[:,:,i] = filter_avg_naive(imgAux[:,:,i], HEIGHT, WIDTH)
	# img[:,:,i] = filter_avg_integral_img(imgAux[:,:,i], HEIGHT, WIDTH)


# img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)

aux = np.zeros(img.shape, dtype=np.ubyte)
aux = img*255
img = aux

cv.imwrite(IMG_OUT, img)
# cv.namedWindow(IMG_OUT, cv.WINDOW_NORMAL)
# cv.imshow(IMG_OUT, img)
# cv.waitKey(0)
# cv.destroyAllWindows()
