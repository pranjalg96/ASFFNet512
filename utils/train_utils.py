import cv2
import numpy as np
import matplotlib.pyplot as plt


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

