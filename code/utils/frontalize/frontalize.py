import numpy as np
import cv2
from frontalization.utils import frontalize_landmarks
frontalization_weights = np.load('/srv/storage/datasets/rafaelvieira/text2expression/utils/frontalization/data/frontalization_weights.npy')


def format_kps(kps):

    formatted_kps = list()
    formatted_kps_filtered = list()
    desired_left_eye = (0.4, 0.4)
    desired_right_eyex = 1.0 - desired_left_eye[0]

    for idx, frame in enumerate(kps):
        #you might have to chance indexes here
        face_kps = frame[12:80]
        
        try:
            face_kps = frontalize_landmarks(np.asarray(face_kps), frontalization_weights)
        except:
            import traceback
            print(traceback.format_exc())
            pass

        left_eye_kps = np.array(face_kps[36:42])
        right_eye_kps = np.array(face_kps[42:48])
        left_eye_center = left_eye_kps.mean(axis = 0).astype("int")
        right_eye_center = right_eye_kps.mean(axis = 0).astype("int")

        dy = right_eye_center[1] - left_eye_center[1]
        dx = right_eye_center[0] - left_eye_center[0]
        angle = np.degrees(np.arctan2(dy, dx))
        dist = np.sqrt((dx ** 2) + (dy ** 2))
        
        if dist == 0:
            continue
        desired_dist = (desired_right_eyex - desired_left_eye[0])
        desired_dist *= 256
        scale = desired_dist/dist
        eyes_center = ((left_eye_center[0] + right_eye_center[0]) // 2, (left_eye_center[1] + right_eye_center[1]) // 2)
        
        M = cv2.getRotationMatrix2D((int(eyes_center[0]), int(eyes_center[1])), angle, scale)
        
        tX = 256 * 0.5
        tY = 256 * desired_left_eye[1]
        M[0, 2] += (tX - eyes_center[0])
        M[1, 2] += (tY - eyes_center[1])
        (w, h) = (256, 256)

        ones = np.ones(shape=(len(face_kps), 1))
        points_ones = np.hstack([face_kps, ones])
        transformed_points = M.dot(points_ones.T).T
        transformed_points = transformed_points/256
        #transformed_points = remove_kps(transformed_points)

        up_mid_mouth = transformed_points[62]
        dw_mid_mouth = transformed_points[66]
        center = (up_mid_mouth + dw_mid_mouth)/2
        center = np.expand_dims(center, axis = 0)
        transformed_points = np.concatenate((transformed_points, center), axis = 0)
        formatted_kps.append(transformed_points)

    formatted_kps = np.asarray(formatted_kps)
    #formatted_kps_nf = copy.deepcopy(formatted_kps)

    return formatted_kps.tolist()