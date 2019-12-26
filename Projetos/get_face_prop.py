import cv2 as cv
import dlib
import numpy as np
import pprint
import math 
import imutils

pp = pprint.PrettyPrinter(indent=4)

filename = "teste12.jpg"
filename_write = "teste12_save.bmp"
filename_predictor = "./shape_predictor_68_face_landmarks.dat"

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(filename_predictor)

WHITE_EYE_MIN = (180, 180, 180)
BLACK_EYE_MAX = (70, 70, 70)

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


def get_color_eyes(face_dict, image):
    # Coordinate most to the left of the eye and most to the right
    eye_left_min = face_dict["left_eye"]["coords"][0] # landmark 37
    eye_left_max = face_dict["left_eye"]["coords"][3] # landmark 40
    
    img_hls = cv.cvtColor(image, cv.COLOR_BGR2HLS)

    # line parameters y = ca*x+cl (angular and linear coeficient)
    ca = (eye_left_max[1] - eye_left_min[1]) / (eye_left_max[0] - eye_left_min[0])
    cl = eye_left_min[1] - ca*eye_left_min[0]

    pixels = []
    r_acm, g_acm, b_acm = 0, 0, 0
    diff = int((eye_left_max[0] - eye_left_min[0])*0.3)

    for x in range(eye_left_min[0]+diff, eye_left_max[0]-diff): # take off skin
        y = int(ca*x+cl)
        '''
        r, g, b = image[y, x, 0], image[y, x, 1], image[y, x, 2]
        print(r, g, b)
        # if its not black or white
        if((r < WHITE_EYE_MIN[0] or g < WHITE_EYE_MIN[1] or b < WHITE_EYE_MIN[2]) and \
           (r > BLACK_EYE_MAX[0] or g > BLACK_EYE_MAX[1] or b > BLACK_EYE_MAX[2])):
           pixels.append([x, y])
           r_acm += r
           g_acm += g
           b_acm += b
        '''
        h, l, s = img_hls[y, x, 0], img_hls[y, x, 1], img_hls[y, x, 2]
        # print(h, l, s)
        # saturation and luminence check or if its green/blue
        if((l < 150 and l > 30 and s > 20) or (h > 50 and h < 110)):
            pixels.append([x, y])
            r_acm += image[y, x, 2]
            g_acm += image[y, x, 1]
            b_acm += image[y, x, 0]

    #for pixel in pixels:
    #    cv.circle(image_save, (pixel[0], pixel[1]), 1, (0, 0, 255), -1)

    r_avg, g_avg, b_avg = r_acm/len(pixels), g_acm/len(pixels), b_acm/len(pixels)
    # cv.circle(image_save, (50,50), 25, (b_avg, g_avg, r_avg), -1)

    return (r_avg, g_avg, b_avg)


def check_face_rotation(face_dict):
    left_eye_coords = np.array(face_dict["left_eye"]["coords"])
    right_eye_coords = np.array(face_dict["right_eye"]["coords"])

    x0 = np.average(left_eye_coords[:,0])
    y0 = np.average(left_eye_coords[:,1])

    x1 = np.average(right_eye_coords[:,0])
    y1 = np.average(right_eye_coords[:,1])

    angle = math.atan2(y1-y0, x1-x0)
    angle *= 180/math.pi
    # print(angle)
    return -angle # sign is inverted (y inverted)


def rotate_image(angle, image):
    rotated = imutils.rotate_bound(image, angle)
    return rotated


def get_face_dict(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # detect faces in the grayscale image
    rects = detector(gray, 1)

    face = dict()
    r = rects[0]
    rect_face = [[r.left(), r.top()],
        [r.right(), r.top()],
        [r.left(), r.bottom()],
        [r.right(), r.bottom()]]

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

    return face


def get_mouth_prop(face_dict, image):
    point_left = face_dict["mouth"]["coords"][0] # landmark 49
    point_lower = face_dict["mouth"]["coords"][-2] # landmark 67
    point_right = face_dict["mouth"]["coords"][6] # landmark 55
    point_higher = face_dict["mouth"]["coords"][-6] # landmark 63
    
    # landmark 63 y - landmark 52 y
    lips_height = point_higher[1] - face_dict["mouth"]["coords"][3][1]

    # Average height difference between lips and the most left and most right mouth point
    # inverted because y axis grows down
    avg_diff_higher = ((point_left[1]-point_higher[1]) + (point_right[1]-point_higher[1])) / 2
    avg_diff_lower = ((point_lower[1]-point_left[1]) + (point_lower[1]-point_right[1])) / 2

    state = str()
    # 3 possible states: smiling, normal, scared
    if(avg_diff_lower <= lips_height):
        state = "normal"
    elif(avg_diff_lower > lips_height and avg_diff_higher < lips_height):
        state = "smiling"
    else:
        state = "scared"
    # print(state)
    return state


def get_skin_color(face_dict, image):
    nose_point1 = face_dict["nose_high"]["coords"][0] # landmark 28
    nose_point2 = face_dict["nose_low"]["coords"][0] # landmark 32
    nose_point3 = face_dict["nose_low"]["coords"][-1] # landmark 36

    eyebrown_point1 = face_dict["left_eyebrown"]["coords"][2] # landmark 20
    eyebrown_point2 = face_dict["right_eyebrown"]["coords"][2] # landmark 25

    # rect = [x0, y0, x1, y1]
    # rectangle on nose
    rect_nose = [nose_point2[0], nose_point1[1], nose_point3[0], \
        nose_point3[1]]
    rect_forehead = [eyebrown_point1[0], eyebrown_point1[1] - int(0.5*(rect_nose[3]-rect_nose[1])),\
        eyebrown_point2[0], eyebrown_point1[1] - int(0.1*(rect_nose[3]-rect_nose[1]))]
    
    r_acm, g_acm, b_acm = 0, 0, 0
    for y in range(rect_nose[1], rect_nose[3]+1):
        for x in range(rect_nose[0], rect_nose[2]+1):
            r_acm += image[y, x, 2]
            g_acm += image[y, x, 1]
            b_acm += image[y, x, 0]

    for y in range(rect_forehead[1], rect_forehead[3]+1):
        for x in range(rect_forehead[0], rect_forehead[2]+1):
            r_acm += image[y, x, 2]
            g_acm += image[y, x, 1]
            b_acm += image[y, x, 0]

    cv.rectangle(image_save, tuple(rect_forehead[:2]), tuple(rect_forehead[-2:]), color=(0,255,0), thickness=5)
    cv.rectangle(image_save, tuple(rect_nose[:2]), tuple(rect_nose[-2:]), color=(0,0,255), thickness=5)

    total_pixels = (rect_nose[2]-rect_nose[0])*(rect_nose[3]-rect_nose[1])
    total_pixels += (rect_forehead[2]-rect_forehead[0])*(rect_forehead[3]-rect_forehead[1])

    r_avg = r_acm / total_pixels
    g_avg = g_acm / total_pixels
    b_avg = b_acm / total_pixels
    # print(r_avg, g_avg, b_avg)
    cv.circle(image_save, (100,50), 25, (b_avg, g_avg, r_avg), -1)

    return (r_avg, g_avg, b_avg)


def get_face_info(filename):
    global image_save

    image_save = cv.imread(filename, cv.IMREAD_COLOR)
    image_face = cv.imread(filename, cv.IMREAD_COLOR)

    face = get_face_dict(image_face)
    angle = check_face_rotation(face)
    if(angle > 5 or angle < -5):
        image_face = rotate_image(angle, image_face)
        image_save = rotate_image(angle, image_save)
        face = get_face_dict(image_face)
        for p in ["left_eye","right_eye"]:
            for (x, y) in face[p]["coords"]:
                cv.circle(image_face, (x, y), 5, face[p]["color"], -1)
        cv.imwrite(filename.split(".")[-2]+"-putin.bmp", image_face)

    color_eyes = get_color_eyes(face, image_face)
    mouth_state = get_mouth_prop(face, image_face)
    skin_color = get_skin_color(face, image_face)

    #for p in face:
    #    for (x, y) in face[p]["coords"]:
    #        cv.circle(image_save, (x, y), 5, face[p]["color"], -1)

    cv.imwrite(filename.split(".")[-2]+"-info.bmp", image_save)

    return {
        "eyes_color": color_eyes,
        "mouth_state": mouth_state,
        "skin_color": skin_color
    }

image_save = None

if(__name__ == "__main__"):
    # load image from file
    image_face = cv.imread(filename, cv.IMREAD_COLOR)
    image_save = cv.imread(filename, cv.IMREAD_COLOR)
    face = get_face_dict(image_face)
    angle = check_face_rotation(face)
    if(angle > 5 or angle < -5):
        image_face = rotate_image(angle, image_face)
        image_save = rotate_image(angle, image_save)
        face = get_face_dict(image_face)

    color_eyes = get_color_eyes(face, image_face)
    mouth_state = get_mouth_prop(face, image_face)
    skin_color = get_skin_color(face, image_face)

    for p in face:
        for (x, y) in face[p]["coords"]:
            cv.circle(image_face, (x, y), 5, face[p]["color"], -1)

    cv.imwrite(filename_write, image_save)

    print(filename)
