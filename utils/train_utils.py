import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import math

def get_lmarks_from_tensor(image, FaceDetection):
    """
    Given a processed image tensor, get the landmarks from it
    :param: image - A processed image tensor
    :param: FaceDetection - Face Detection module

    returns 
    face - Landmarks in the image tensor, None if no face detected
    """
    image = image.permute(1, 2, 0)
    image = (image + 1) * 127.5
    image = image.numpy().astype('uint8')
    
    # Get landmarks in image
    faces = FaceDetection.get_landmarks(image)

    # No face detected
    if faces is None:
        return None
    
    i = 0
    # More than one face found, only get the landmarks of largest one
    if len(faces) > 1:
        sizes = []
        for lmarks  in faces:
            sizes.append(lmarks[8, 1] - lmarks[19, 1])

        i = np.argmax(sizes)

    # Select the largest face (or the only one)
    face = faces[i]

    return face


def generate_lq_point_mask(lq_landmarks):
    PointMask = torch.zeros((1, 512, 512))
    for i in range(17, len(lq_landmarks)):
        point_x = lq_landmarks[i][0]
        point_y = lq_landmarks[i][1]
        if point_x > 1 and point_y > 1 and point_x < 512 - 2 and point_y < 512 - 2:
            PointMask[0,int(math.floor(point_y))-3:int(math.ceil(point_y))+3,int(math.floor(point_x))-3:int(math.ceil(point_x))+3] = 1
            
    return PointMask.unsqueeze(0)

