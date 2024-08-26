import os
import sys
import numpy as np
import codecs
import cv2
import uuid
from frontalization.utils import frontalize_landmarks
import warnings
warnings.filterwarnings("error")
X_MAX_DIM = 1280
Y_MAX_DIM = 720
frontalization_weights = np.load('frontalization/data/frontalization_weights.npy')
def read_data(file_name):
    info = np.load(file_name, allow_pickle=True)
    dataset = info['dataset'][()]
    vocabulary = info['vocabulary']
    return dataset, vocabulary

def sample_data(dataset, percentage = 1):
    sampled_data = dict()
    keys = list(dataset.keys())
    np.random.seed(42)
    sample_size = int(len(keys)*percentage)
    dataset_indices = np.random.choice(len(keys), size = sample_size, replace = False)

    for idx in dataset_indices:
        selected_key = keys[idx]
        selected_sample = dataset[selected_key]
        sampled_data[selected_key] = selected_sample
    
    return sampled_data

def format_kps(kps):

    formatted_kps = list()
    formatted_kps_filtered = list()
    desired_left_eye = (0.4, 0.4)
    desired_right_eyex = 1.0 - desired_left_eye[0]

    for idx, frame in enumerate(kps):
        face_kps = frame[12:80]

        try:
            face_kps = frontalize_landmarks(np.asarray(face_kps), frontalization_weights)
        except:
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
        formatted_kps.append(transformed_points)

    formatted_kps = np.asarray(formatted_kps)
    #formatted_kps_nf = copy.deepcopy(formatted_kps)

    return formatted_kps.tolist()

def draw_keypoints(kps, dims = (256, 256)):
    img = np.ones(dims) * 255
    for kp in kps:
        x = kp[0]
        y = kp[1]
        img = cv2.circle(img, (int(x), int(y)), 2, (0, 255 ,0), 2)
    return img

def write_dataset(flame_dataset, phase, out_root):
    print("Writing data")
    files = list()
    texts = list()
    parameters = list()
    landmarks = list()
    cams = list()

    for data in flame_dataset:
        files.append(data[0])
        texts.append(data[1])
        parameters.append(data[2])
    
    files_fp = os.path.join(out_root, "{}.files".format(phase))
    skels_fp = os.path.join(out_root, "{}.skels".format(phase))
    texts_fp = os.path.join(out_root, "{}.text".format(phase))

    with codecs.open(files_fp, "w", "utf-8") as f:
        for file_n in files:
            f.write(file_n)
            f.write("\n")
        f.close()

    with codecs.open(texts_fp, "w", "utf-8") as f:
        for text in texts:
            f.write(text)
            f.write("\n")
        f.close()


    with open(skels_fp, "w") as f:
        for param in parameters:
            for idx, ps in enumerate(param):
                for p in ps:
                    f.write(str(p[0]) + " " + str(p[1]) + " ")
                try:
                    f.write(str(idx/(len(param) - 1)) + " ")
                except:
                    import pdb
                    pdb.set_trace()

            f.write("\n")
        f.close()

def flatten_params(pose, expression, shape):
    flattened = list()
    for p, e, s in zip(pose, expression, shape):
        flattened.append(e.tolist() + s.tolist() + p.tolist())
    return flattened

def flatten_landmarks(landmarks):
    landmarks_flattened = list()
    for landmark in landmarks:
        landmarks_flattened.append(landmark.flatten().tolist())
    return landmarks_flattened

def flatten_cams(cams):
    cams_flattened = list()
    for cam in cams:
        cams_flattened.append(cam.tolist())
    return cams_flattened

def main():

    phase = sys.argv[1]
    sample_path = "/home/rafaelv/masters/datasets/how2sign/validation_sample.npz"
    out_root = "/home/rafaelv/masters/ProgressiveTransformersSLP/Data_scaled5/tmp/"
    os.makedirs(out_root, exist_ok = True)
    dataset, vocabulary = read_data(sample_path)

    if phase == "train":
        allow_list = os.listdir("/home/rafaelv/masters/datasets/slp_dataset_f68_full_ps_new/slp_dataset_f68_full_ps/train")
    else:
        allow_list = os.listdir("/home/rafaelv/masters/datasets/slp_dataset_f68_full_ps_new/slp_dataset_f68_full_ps/val")

    full_dataset = list()

    allow_list = [a.replace(".pkl", "") for a in allow_list]
    for idx, (video_name, data) in enumerate(dataset.items()):
        print("Running video idx: {}".format(idx))
        if video_name in allow_list:
            text = data['text']
            kps = data['kp']
            kps_formatted = format_kps(kps)
            full_dataset.append((os.path.join(phase, video_name), text, kps_formatted))
    
    write_dataset(full_dataset, phase, out_root)
    




if __name__ == "__main__":
    main()