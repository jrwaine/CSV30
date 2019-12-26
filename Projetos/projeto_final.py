import cv2 as cv
import dlib
import numpy as np

filename = "teste.jpg"
filename_write = "teste_save.bmp"
predictor = "./shape_predictor_68_face_landmarks.dat"


def rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
 
    # return a tuple of (x, y, w, h)
    return (x, y, w, h)


def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)
 
    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
 
    # return the list of (x, y)-coordinates
    return coords


def detect_landmarks(face_rect, gray):
    # determine the facial landmarks for the face region, then
    # convert the facial landmark (x, y)-coordinates to a NumPy
    # array
    shape = predictor(gray, face_rect)
    shape = shape_to_np(shape)

    return shape


# load image from file
image_face = cv.imread(filename, cv.IMREAD_COLOR)
gray = cv.cvtColor(image_face, cv.COLOR_BGR2GRAY)

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor)

# detect faces in the grayscale image
rects = detector(gray, 1)

face = dict()
r = rects[0]
rect_face = [[r.top(), r.left()],
    [r.top(), r.right()],
    [r.bottom(), r.left()],
    [r.bottom(), r.right()]]
print(rect_face)
face["rect"] = {"coords": np.array(rect_face), "color": (50, 127, 127)}

landmarks = detect_landmarks(rects[0], gray)


face["chin"]            = {"coords":  landmarks[0:17], "color": (0, 0, 255)}
face["left_eyebrown"]   = {"coords": landmarks[17:22], "color": (0, 255, 0)}
face["right_eyebrown"]  = {"coords": landmarks[22:27], "color": (255, 0, 0)}
face["nose_high"]       = {"coords": landmarks[27:31], "color": (0, 255, 255)}
face["nose_low"]        = {"coords": landmarks[31:36], "color": (255, 0, 255)}
face["left_eye"]        = {"coords": landmarks[36:42], "color": (255, 255, 0)}
face["right_eye"]       = {"coords": landmarks[42:48], "color": (255, 255, 255)}
face["mouth"]           = {"coords": landmarks[48:-1], "color": (0, 0, 0)}

for p in face:
    for (x, y) in face[p]["coords"]:
        cv.circle(image_face, (x, y), 5, face[p]["color"], -1)



cv.imwrite(filename_write, image_face)



