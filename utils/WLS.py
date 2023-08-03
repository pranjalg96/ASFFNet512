import numpy as np

def calc_sim(lq_landmarks, ref_landmark):
    weight_eye = 1.01
    weight_mouth = 1.49

    # lq landmark process    (68,2)
    lq_landmarks_eye = np.concatenate([lq_landmarks[17:27, :], lq_landmarks[36:48, :]], axis=0)  # eye+eyebrow landmark

    # ref landmark process
    Ref_Ab = np.insert(ref_landmark, 2, 1, -1)
    Ref_Ab_eye = np.concatenate([Ref_Ab[17:27, :], Ref_Ab[36:48, :]], axis=0)  # eyebrow+eye (22, 3)
    Ref_Ab_mouth = Ref_Ab[48:, :]  # mouth #(20,2)
    

    # eye
    result_Ab_eye = np.dot(np.dot(np.linalg.inv(np.dot(Ref_Ab_eye.T, Ref_Ab_eye)), Ref_Ab_eye.T), lq_landmarks_eye) #(3, 2)
    # mouth
    result_Ab_mouth = np.dot(np.dot(np.linalg.inv(np.dot(Ref_Ab_mouth.T, Ref_Ab_mouth)), Ref_Ab_mouth.T), lq_landmarks[48:, :]) #(3, 2)

    ref_landmark_align_eye = np.dot(Ref_Ab_eye, result_Ab_eye.reshape([3, 2]))  # transposed eye landmark (22, 2)
    ref_landmark_align_mouth = np.dot(Ref_Ab_mouth, result_Ab_mouth.reshape([3, 2]))  # transposed mouth landmark (20, 2)
    Sim = weight_eye * np.linalg.norm(ref_landmark_align_eye - lq_landmarks_eye) + weight_mouth * np.linalg.norm(ref_landmark_align_mouth - lq_landmarks[48:, :])
    return Sim


def guidance_selection(lq_lmarks, hq_lmarks):
    """
    Guidance selection algorithm according to inference code
    :param: lq_lmarks - Numpy array of Low-quality image landmarks
    :param: hq_lmarks - Tensor of High-quality landmarks of images

    returns
    hq_selected_idx - Index of the selected high-quality image and landmarks
    """
    hq_lmarks = hq_lmarks.numpy()

    similarities = []
    for i in range(len(hq_lmarks)):
        hq_lmark = hq_lmarks[i]
        similarity = calc_sim(lq_landmarks=lq_lmarks, ref_landmark=hq_lmark)

        similarities.append(similarity)

    return np.argmin(similarities)
