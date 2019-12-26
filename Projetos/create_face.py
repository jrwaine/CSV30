import cv2 as cv
import numpy as np
import imutils
from get_face_prop import get_face_info

# height, width
IMAGE_SIZE = (628, 600)
size_img = [IMAGE_SIZE[0], IMAGE_SIZE[1], 4] # bgra

images_test = [
    {
        "filename": "teste12.jpg",
        "color_hair": "black",
        "face_format": "diamond",
        "sex": "female"
    },
    {
        "filename": "teste15.jpg",
        "color_hair": "brown",
        "face_format": "diamond",
        "sex": "male"
    },
    {
        "filename": "teste8.jpg",
        "color_hair": "white",
        "face_format": "square",
        "sex": "male"
    },
    {
        "filename": "teste.jpg",
        "color_hair": "brown",
        "face_format": "diamond",
        "sex": "male"
    },
    {
        "filename": "teste1.jpeg",
        "color_hair": "red",
        "face_format": "square",
        "sex": "female"
    },
    {
        "filename": "teste3.jpeg",
        "color_hair": "red",
        "face_format": "oval",
        "sex": "female"
    },
    {
        "filename": "teste4.jpg",
        "color_hair": "brown",
        "face_format": "diamond",
        "sex": "female"
    },
    {
        "filename": "teste7.png",
        "color_hair": "black",
        "face_format": "oval",
        "sex": "female"
    },
    {
        "filename": "teste9.jpg",
        "color_hair": "blonde",
        "face_format": "diamond",
        "sex": "male"
    },
    {
        "filename": "teste11.jpg",
        "color_hair": "brown",
        "face_format": "oval",
        "sex": "male"
    },
    {
        "filename": "teste13.jpg",
        "color_hair": "white",
        "face_format": "oval",
        "sex": "male"
    }
]

sizes = {
    # x, y
    "mouth":(175, 150),
    "male_hair":(600, 450),
    "female_hair":(550, 10000), # better be per width
    "eyes":(400, 300),
    "face":(600, 628),
    "nose":(60, 15)
}

positions = {
    # central position x, y
    "mouth": (300, 525),
    "male_hair": (300, 210),
    "female_hair":(300, 300),
    "eyes": (300, 325),
    "face": (300, 314),
    "nose": (300, 450)
}


hair_colors = {
    "brown": (103, 74, 30),
    "blue": (7, 67, 233),
    "red": (230, 25, 15),
    "green": (22, 171, 3),
    "blonde": (229, 222, 11),
    "black": (20, 20, 20),
    "white": (230, 220, 200)
}


filenames = {
    "mouth": {
        "normal": "images_template/normal_mouth.png",
        "smiling":"images_template/opened_mouth.png",
        "scared":"images_template/scared_mouth.png"
    },
    "hair": {
        "male": "images_template/male_hair_front1.png",
        "female": "images_template/female_hair.png"
    },
    "eyes": {
        "normal": "images_template/eyes.png"
    },
    "face":{
        "oval": "images_template/oval_face.png",
        "square": "images_template/square_face.png",
        "diamond": "images_template/diamond_face.png"
    },
    "nose":{
        "normal": "images_template/nose5.png"
    }
}


def put_image_over_another(img_original, img_to_put, coord_center):
    coord = get_initial_coord(img_to_put, coord_center)
    for y in range(max(0, coord[1]), coord[1]+img_to_put.shape[0]):
        for x in range(max(0, coord[0]), coord[0]+img_to_put.shape[1]):
            for i in range(0, 3):
                x_put, y_put = x-coord[0], y-coord[1]
                alpha = img_to_put[y_put, x_put, 3]/255
                img_original[y, x, i] = int(img_original[y, x, i] * (1-alpha) + img_to_put[y_put, x_put, i]*alpha)
    return img_original


def get_initial_coord(img, center_coord):
    return (center_coord[0]-img.shape[1]//2, center_coord[1]-img.shape[0]//2)


def import_face(face_filename):
    face = cv.imread(face_filename, cv.IMREAD_UNCHANGED)
    ratio = min(sizes["face"][1]/face.shape[0], sizes["face"][0]/face.shape[1])
    face = imutils.resize(face, width=int(face.shape[1]*ratio))
    return face


def import_mouth(mouth_filename):
    mouth = cv.imread(mouth_filename, cv.IMREAD_UNCHANGED)
    ratio = min(sizes["mouth"][1]/mouth.shape[0], sizes["mouth"][0]/mouth.shape[1])
    mouth = imutils.resize(mouth, width=int(mouth.shape[1]*ratio))
    return mouth


def import_hair(hair_filename, hair_sex):
    hair = cv.imread(hair_filename, cv.IMREAD_UNCHANGED)
    ratio = min(sizes[hair_sex][1]/hair.shape[0], sizes[hair_sex][0]/hair.shape[1])
    hair = imutils.resize(hair, width=int(hair.shape[1]*ratio))
    return hair


def import_eyes(eyes_filename):
    eyes = cv.imread(eyes_filename, cv.IMREAD_UNCHANGED)
    ratio = min(sizes["eyes"][1]/eyes.shape[0], sizes["eyes"][0]/eyes.shape[1])
    eyes = imutils.resize(eyes, width=int(eyes.shape[1]*ratio))
    return eyes


def import_nose(nose_filename):
    nose = cv.imread(nose_filename, cv.IMREAD_UNCHANGED)
    ratio = min(sizes["nose"][1]/nose.shape[0], sizes["nose"][0]/nose.shape[1])
    nose = imutils.resize(nose, width=int(nose.shape[1]*ratio))
    return nose


def paint_eyes(eyes_image, color):
    value_min = 30
    value_max = 230
    aux = np.zeros((1, 1, 3), dtype='uint8')

    aux[0,0,0] = color[0]
    aux[0,0,1] = color[1]
    aux[0,0,2] = color[2]

    r, g, b = eyes_image[:,:,2],eyes_image[:,:,1],eyes_image[:,:,0]

    h_color = cv.cvtColor(aux, cv.COLOR_RGB2HLS)
    h_color = h_color[0,0]
    mask = np.where(np.logical_and(np.logical_and(r > value_min, g > value_min, b > value_min), \
        np.logical_and(g < value_max, b < value_max)), 1, 0)

    img_hls = cv.cvtColor(eyes_image, cv.COLOR_BGR2HLS)
    h = img_hls[:,:,0]
    l = img_hls[:,:,1]
    s = img_hls[:,:,2]
    img_hls[:,:,0] = np.where(mask, h_color[0], h)
    img_hls[:,:,1] = np.where(mask, l+20, l)
    img_hls[:,:,2] = np.where(mask, h_color[2], s)

    aux = cv.cvtColor(img_hls, cv.COLOR_HLS2BGR)
    eyes_image[:,:,0] = aux[:,:,0]
    eyes_image[:,:,1] = aux[:,:,1]
    eyes_image[:,:,2] = aux[:,:,2]
    return eyes_image


def paint_skin(face_image, color):
    value_min = 50
    r, g, b = face_image[:,:,2],face_image[:,:,1],face_image[:,:,0]
    mask = np.where(np.logical_and(r > value_min, g > value_min, b > value_min), 1, 0)
    face_image[:,:,0] = np.where(mask, color[2], 0)
    face_image[:,:,1] = np.where(mask, color[1], 0)
    face_image[:,:,2] = np.where(mask, color[0], 0)

    return face_image


def paint_hair(hair_image, color):
    value_min = 220
    r, g, b = hair_image[:,:,2], hair_image[:,:,1], hair_image[:,:,0]
    mask = np.where(b > value_min, 1, 0)
    hair_image[:,:,0] = np.where(mask, color[2], 0)
    hair_image[:,:,1] = np.where(mask, color[1], 0)
    hair_image[:,:,2] = np.where(mask, color[0], 0)

    return hair_image


def process_face(dict_info_image):
    drawed_face = np.zeros(tuple(size_img))
    dict_info = get_face_info(dict_info_image["filename"])

    eyes = import_eyes(filenames["eyes"]["normal"])
    eyes = paint_eyes(eyes, dict_info["eyes_color"])

    face = import_face(filenames["face"][dict_info_image["face_format"]])
    face = paint_skin(face, dict_info["skin_color"])

    if(dict_info["mouth_state"] == "smiling"):
        mouth = import_mouth(filenames["mouth"]["smiling"])
    elif(dict_info["mouth_state"] == "scared"):
        mouth = import_mouth(filenames["mouth"]["scared"])
    else:
        mouth = import_mouth(filenames["mouth"]["normal"])

    if(dict_info_image["color_hair"] != "none"):
        if(dict_info_image["sex"] == "female"):
            hair = import_hair(filenames["hair"][dict_info_image["sex"]], "female_hair")
        else:
            hair = import_hair(filenames["hair"][dict_info_image["sex"]], "male_hair")
        paint_hair(hair, hair_colors[dict_info_image["color_hair"]])

    nose = import_nose(filenames["nose"]["normal"])

    drawed_face = put_image_over_another(drawed_face, face, positions["face"])
    drawed_face = put_image_over_another(drawed_face, mouth, positions["mouth"])
    drawed_face = put_image_over_another(drawed_face, eyes, positions["eyes"])
    if(dict_info_image["color_hair"] != "none"):
        if(dict_info_image["sex"] == "female"):
            drawed_face = put_image_over_another(drawed_face, hair, positions["female_hair"])
        else:
            drawed_face = put_image_over_another(drawed_face, hair, positions["male_hair"])
    drawed_face = put_image_over_another(drawed_face, nose, positions["nose"])

    # img_save = cv.cvtColor(img_masked, cv.COLOR_RGBA2RGB)
    cv.imwrite(dict_info_image["filename"].split(".")[-2]+"-drawed.bmp", drawed_face)
    print("processou", dict_info_image["filename"])


if(__name__ == "__main__"):
    for i in images_test:
        process_face(i)
