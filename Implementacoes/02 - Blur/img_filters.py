import cv2 as cv
import numpy as np



# img is a 2d numpy array normalized in [0, 1]
# h is height and w width
def filter_avg_naive(img, h, w):
    x_range = range(w//2, img.shape[1]-w//2)
    y_range = range(h//2, img.shape[0]-h//2)
    area = h*w
    img_aux = np.zeros(img.shape)
    for y in y_range:
        for x in x_range:
            sum = 0
            for yp in range(y-h//2, y+h//2+1):
                for xp in range(x-w//2, x+w//2+1):
                    sum += img[yp,xp]
            img_aux[y,x] = sum/area
    return img_aux

def filter_avg_separated(img, h, w):
    img_h = np.zeros(img.shape)
    img_aux = np.zeros(img.shape)
    x_range = range(w//2, img.shape[1]-w//2)
    y_range = range(h//2, img.shape[0]-h//2)
    for y in y_range:
        for x in x_range:
            sum = 0
            for xp in range(x-w//2, x+w//2+1):
                sum += img[y,xp]
            img_h[y, x] = sum/w
    for y in y_range:
        for x in x_range:
            sum = 0
            for yp in range(y-h//2, y+h//2+1):
                sum += img_h[yp,x]
            img_aux[y, x] = sum/h
    return img_aux

def filter_avg_integral_img(img, h, w):
    img_integral = np.zeros(img.shape)
    img_aux = np.zeros(img.shape)
    y_range = range(0, img.shape[0])
    x_range = range(0, img.shape[1])

    for y in range(0, img.shape[0]):
        for x in range(0, img.shape[1]):
            img_integral[y, x] = img[y, x]
            if(x > 0):
                img_integral[y, x] += img_integral[y, x-1]
            if(y > 0):
                img_integral[y, x] += img_integral[y-1, x]
            if(x > 0 and y > 0):
                img_integral[y, x] -= img_integral[y-1, x-1]
    
    for y in y_range:
        for x in x_range:
            ymin = y-h//2-1
            ymax = y+h//2
            xmin = x-w//2-1
            xmax = x+h//2
            sum = 0
            sum += img_integral[min(y_range[-1], ymax), min(x_range[-1], xmax)]
            if(ymin >= 0):
                sum -= img_integral[ymin, min(x_range[-1], xmax)]
            if(xmin >= 0):
                sum -= img_integral[min(y_range[-1], ymax), xmin]
            if(ymin >= 0 and xmin >= 0):
                sum += img_integral[ymin, xmin]
            yr = min(ymax, y_range[-1])-max(ymin, -1)
            xr = min(xmax, x_range[-1])-max(xmin, -1)
            area = max(1,yr)*max(1,xr)
            img_aux[y,x] = sum/area
    return img_aux

