from itertools import starmap

import dlib
import numpy as np
from scipy.integrate import simpson

from . import settings


def seed_everything(seed: int):
    import os
    import random

    import numpy as np
    import torch

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def load_pts(path):
    """
    takes as input the path to a .pts and returns a list of 
	tuples of floats containing the points in in the form:
	[(x_0, y_0, z_0),
	 (x_1, y_1, z_1),
	 ...
	 (x_n, y_n, z_n)]"""
    with open(path) as f:
        rows = [rows.strip() for rows in f]
    
    """Use the curly braces to find the start and end of the point data""" 
    head = rows.index('{') + 1
    tail = rows.index('}')

    """Select the point data split into coordinates"""
    raw_points = rows[head:tail]
    coords_set = [point.split() for point in raw_points]

    """Convert entries from lists of strings to tuples of floats"""
    points = [[float(point) for point in coords] for coords in coords_set]
    return points


def get_rectangle(image_path, landmarks):
    face_detector = dlib.cnn_face_detection_model_v1(settings.FACE_DETECTOR_MODEL_PATH)

    img = dlib.load_rgb_image(image_path)

    # надо найти тот прямоугольник, который соотествует real_landmarks, если на изображении несколько прямоугольников
    rect, point_in_rect, n_faces = find_true_rectangle(img, landmarks, face_detector)

    rect_x1 = rect[0]
    rect_y1 = rect[1]
    rect_x2 = rect[2]
    rect_y2 = rect[3]
    quantity_point = point_in_rect
    quantity_faces = n_faces

    return rect_x1, rect_y1, rect_x2, rect_y2, quantity_point, quantity_faces


def calculate_point_in_rect(box,landmarks):
    x1 = box[0]
    y1 = box[1]
    x2 = box[2]
    y2 = box[3]
    points = 0
    for point in landmarks:
        if (x1<point[0]<x2) and (y1<point[1]<y2):
            points+= 1 
    
    return(points)


def find_true_rectangle(img, landmarks, face_detector):
    detections = face_detector(img, 1)

    # создаем список боксов детекции для каждой картинки
    if len(detections) == 0:
        rect = [None, None, None, None]
        point_in_rect = None
    elif len(detections) >= 1:
        face_coords = [] # координаты боксов для детекций лица
        for face in detections: 
            rect = face.rect

            x1 = rect.left()
            y1 = rect.top()
            x2 = rect.right()
            y2 = rect.bottom()
                
            face_coords.append([x1,y1,x2,y2])

        if len(face_coords)==1:
            rect = face_coords[0]
            point_in_rect = calculate_point_in_rect(rect,landmarks)
        else:
            point_in_rect = []
            for box in face_coords:
                point_in_rect.append(calculate_point_in_rect(box,landmarks))

            rect = face_coords[np.argmax(point_in_rect)]

            point_in_rect = max(point_in_rect)
        
    return rect, point_in_rect, len(detections)


def upsample_landmarks(image, landmarks, rect):
    image = image.permute(0,2,3,1).numpy() # BxCxHxW -- BxHxWxC
    landmarks = landmarks.numpy() #[batch_size, 68, 2]
    rect = rect.numpy() ##[batch_size, 4]

    #unresize
    unresize_landmarks = list(starmap(unresize, zip(image, landmarks, rect))) # batch_size x [68, 2]

    # print(unresize_landmarks[0].shape)

    uncropped_landmarks = list(starmap(uncrop, zip(unresize_landmarks, rect)))
    # print( uncropped_landmarks[0].shape, len(uncropped_landmarks), uncropped_landmarks)

    landmarks = np.array(uncropped_landmarks)
    
    return landmarks


def unresize(image, landmarks, rect):

    rect_size = (rect[3]-rect[1],rect[2]-rect[0]) #HXW

    scale_x = image.shape[1] / rect_size[1]
    scale_y = image.shape[0] / rect_size[0]


    unresized_landmarks = np.zeros((68,2))

    unresized_landmarks[:,0] = landmarks[:,0] / scale_x
    unresized_landmarks[:,1] = landmarks[:,1] / scale_y

    return  unresized_landmarks

def uncrop(cropped_landmarks, rect):

    uncropped_landmarks = np.zeros((68,2))
    uncropped_landmarks[:,0] = cropped_landmarks[:,0] + np.full(len(uncropped_landmarks), rect[0]) 
    uncropped_landmarks[:,1] = cropped_landmarks[:,1] + np.full(len(uncropped_landmarks), rect[1]) 

    return uncropped_landmarks


def count_avg_norm_dist_butch(real_landmarks, upsamle_pred_landmarks, rect):

    avg_norm_dist_butch = np.array(list(starmap(count_avg_norm_dist, zip(upsamle_pred_landmarks, real_landmarks, rect)))) # array

    return avg_norm_dist_butch

def count_avg_norm_dist(upsamle_pred_landmarks, real_landmarks, rect):
    W = abs(rect[2] - rect[0])
    H = abs(rect[3] - rect[1])

    normalization_factor = np.sqrt(H * W)
    n_points = real_landmarks.shape[0]

    dist = np.sqrt((np.square(real_landmarks-upsamle_pred_landmarks)).sum(axis=1))

    avg_norm_dist = np.sum(dist) / (n_points * normalization_factor)

    return avg_norm_dist

def ced_auc(RMSE, thres=0.08, step=0.0001):

    num_data = len(RMSE)
    # print(RMSE)
    coord_x = np.arange(0, thres + step, step)
    coord_y = np.array([np.count_nonzero(RMSE <= x) for x in coord_x]) / float(num_data)

    
    #  https://stackoverflow.com/questions/13320262/calculating-the-area-under-a-curve-given-a-set-of-coordinates-without-knowing-t
    ced_auc = simpson(coord_y,x = coord_x) / thres

    return ced_auc