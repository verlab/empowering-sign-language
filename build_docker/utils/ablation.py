import os
import sys
import numpy as np
import pickle
import pandas as pd
from frontalization.utils import frontalize_landmarks
import cv2
from shapely.geometry import Polygon
from dtw import dtw
from sklearn.metrics import mean_absolute_error
frontalization_weights = np.load('frontalization/data/frontalization_weights.npy')
N_frames = 65

def get_euclidean_distance(pts1, pts2):
    dists = list()
    for pt1, pt2 in zip(pts1, pts2):
        dist = np.linalg.norm(pt1 - pt2)
        dists.append(dist)
    try:
        return sum(dists)/len(dists)
    except:
        import traceback
        print(traceback.format_exc())
        import pdb
        pdb.set_trace()

def get_landmarks_velocity(pts_lst1, pts_lst2):

    diffs1 = list()
    diffs2 = list()

    for idx in range(0, len(pts_lst1) - 1):
        diff1 = pts_lst1[idx + 1] - pts_lst1[idx]
        diff2 = pts_lst2[idx + 1] - pts_lst2[idx]
        diffs1.append(diff1)
        diffs2.append(diff2)

    diffs1 = np.asarray(diffs1)
    diffs2 = np.asarray(diffs2)
    dist = get_euclidean_distance(diffs1, diffs2)

    return dist

def get_jaw_lips(pts):
    jaw = pts[:, 0:17, :,]
    lips = pts[:, 48:, :,]
    return np.concatenate((jaw, lips), axis = 1)

def get_eyebrows(pts):
    eyeb1 = pts[:, 17:22, :,]
    eyeb2 = pts[:, 22:27, :,]
    return np.concatenate((eyeb1, eyeb2), axis = 1)

def get_mouth(pts):
    return pts[:, 48:, :,]

def _read_instance_gt(root, instance_name):
    instance_fp = os.path.join(root, instance_name)
    with open(instance_fp, "rb") as handler:
        instance = pickle.load(handler)
    return instance

def _read_makeittalk_instance(root, instance_name):
    instance_fp = os.path.join(root, instance_name)
    instance = np.load(instance_fp)[:, :, :-1]
    return instance

def _read_ours_instance(root, instance_name):
    instance_fp = os.path.join(root, instance_name)
    instance = np.load(instance_fp)["arr_0"]
    return instance

def format_kps(kps):

    formatted_kps = list()
    formatted_kps_filtered = list()
    desired_left_eye = (0.4, 0.4)
    desired_right_eyex = 1.0 - desired_left_eye[0]

    for idx, face_kps in enumerate(kps):

        try:
            print("frontalizing")
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
        print(dist)
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

def draw_keypoints(kps):
    img = np.ones((256, 256, 3))*255

    for kp in kps:
        x = kp[0]
        y = kp[1]
        img = cv2.circle(img, (int(x*256), int(y*256)), 2, (0, 255 ,0), 2)
    return img

def downsample_frames(frames, N_FRAMES):
    return [frames[int(idx)] for idx in np.linspace(0, len(frames) - 1, num = N_FRAMES)]

def upsample_frames(frames, N_FRAMES):

    idxs = np.linspace(0, len(frames), num = N_FRAMES)[:-1]
    frames_ = list()

    for idx in idxs:
        frames_.append(frames[int(idx)])
    frames_.append(frames[-1])

    return frames_

def sample_frames(frames):
    if len(frames) > N_frames:
        return downsample_frames(frames, N_frames)
    return upsample_frames(frames, N_frames)


def mouth_opening(predicted_ls, reference_ls):
    diffs = list()
    diffs_gt = list()
    for predicted_l, reference_l in zip(predicted_ls, reference_ls):
        points_p = predicted_l[48:61]*256
        points_r = reference_l[48:61]*256
        points_p = [(int(p[0]), int(p[1])) for p in points_p]
        points_r = [(int(p[0]), int(p[1])) for p in points_r]
        polygon_predicted = Polygon(points_p)
        polygon_reference = Polygon(points_r)
        diffs.append(polygon_predicted.area)
        diffs_gt.append(polygon_reference.area)

    return max(diffs)/max(diffs_gt)

def _read_pt_instance(baseline_root, instance_name):
    instance_fp = os.path.join(baseline_root, instance_name)
    instance = np.load(instance_fp)
    return instance

def draw_keypoints(kps):
    img = np.ones((256, 256, 3))*255

    for kp in kps:
        x = kp[0]
        y = kp[1]
        img = cv2.circle(img, (int(x*256), int(y*256)), 2, (0, 255 ,0), 2)
    return img

def get_eyebrows_mdiff(mean_face, kps):
    right_eyeb = mean_face[17:22]
    left_eyeb = mean_face[22:27]
    diffs = list()

    for kp in kps:
        right_eyeb_predicted = kp[17:22]
        left_eyeb_predicted = kp[22:27]
        right_diff = get_euclidean_distance(right_eyeb_predicted, right_eyeb)
        left_diff = get_euclidean_distance(left_eyeb_predicted, left_eyeb)
        diff = (right_diff + left_diff)/2
        diffs.append(diff)
    return sum(diffs)/len(diffs)

def get_mouth_mdiff(mean_face, kps):
    mean_mouth = mean_face[48:61]
    diffs = list()

    for kp in kps:
        mouth_predicted = kp[48:61]

        mouth_diff = get_euclidean_distance(mean_mouth, mouth_predicted)
        #diff = mouth_diff.sum(axis = 0).sum(axis = 0)
        diffs.append(mouth_diff)
    return sum(diffs)/len(diffs)    

def get_jl_mdiff(mean_face, kps):

    mean_jaw = mean_face[0:17]
    mean_lips = mean_face[48:61]
    mean_jl = np.concatenate((mean_jaw, mean_lips), axis = 0)
    diffs = list()
    for kp in kps:
        jaw_predicted = kp[0:17]
        lips_predicted = kp[48:61]
        jl_predicted = np.concatenate((jaw_predicted, lips_predicted), axis = 0)
        jl_diff = get_euclidean_distance(mean_jl, jl_predicted)
        #diff = mouth_diff.sum(axis = 0).sum(axis = 0)
        diffs.append(jl_diff)
    return sum(diffs)/len(diffs)

def get_mdiff(mean_face, kps):
    diffs = list()
    for kp in kps:
        mdiff = get_euclidean_distance(mean_face, kp)
        diffs.append(mdiff)
    return sum(diffs)/len(diffs)    


def replace_mean_face(my_instances):
    my_instances_formatted = list()
    mean_face = np.load("mean_face.npy")[:-1, :]
    flag = False
    for idx, instance in enumerate(my_instances):
        instance_r = instance.reshape(-1, 136)
        for i in range(0, instance_r.shape[0]):
            for j in range(0, instance_r.shape[1]):
                if instance_r[i][j] < 0 or instance_r[i][j] > 1:
                    flag = True
            if flag:
                print("replacing for mean face")
                instance[i] = mean_face
                flag = False
    return my_instances

def get_mae(pred_instances, gt_instances):
    maes = list()
    for pred_instance, gt_instance in zip(pred_instances, gt_instances):
        maes.append(mean_absolute_error(pred_instance, gt_instance))
    return sum(maes)/len(maes)

def main():

    #jaw-lips, eyebrows, mouth, all lmarks --> euclidean distance and velocity
    ours_root = "/home/rafaelv/masters/text2expression/test_kps_ns_e3/"
    ours_wgp_root = "/home/rafaelv/masters/text2expression/test_kps_ns_e6/"
    ours_wsent_root = "/home/rafaelv/masters/text2expression/test_kps_ns_e12/"
    ours_wsem_root = "/home/rafaelv/masters/text2expression/test_kps_ns_e15/"
    gts_root = "/home/rafaelv/masters/datasets/slp_dataset_f68_new2/slp_dataset_f68/val"
    mean_face = np.load("mean_face.npy")
    euclidean_norm = lambda x, y: np.sum(np.abs(x - y))

    gt_instances = list()
    ours_instances = list()
    ours_wgp_instances = list()
    ours_wsent_instances = list()
    ours_wsem_instances = list()

    for instance_name in os.listdir(gts_root):

        try:
            gt_instances.append(np.asarray(_read_instance_gt(gts_root, instance_name)["kps"])[:, :-1, :])
            ours_instances.append(_read_ours_instance(ours_root, instance_name.replace(".pkl", ".npz"))[:, :-1, :])
            ours_wgp_instances.append(_read_ours_instance(ours_wgp_root, instance_name.replace(".pkl", ".npz"))[:, :-1, :])
            ours_wsent_instances.append(_read_ours_instance(ours_wsent_root, instance_name.replace(".pkl", ".npz"))[:, :-1, :])
            ours_wsem_instances.append(_read_ours_instance(ours_wsem_root, instance_name.replace(".pkl", ".npz"))[:, :-1, :])
        except:
            print("found no existing instance, skipping")
            import traceback
            print(traceback.format_exc())
            continue


    ours_instances = replace_mean_face(ours_instances)
    ours_wgp_instances = replace_mean_face(ours_wgp_instances)
    ours_wsent_instances = replace_mean_face(ours_wsent_instances)
    ours_wsem_instances = replace_mean_face(ours_wsem_instances)

    df = dict()
    metrics_dict = dict(
        ours = (gt_instances, ours_instances),
        ours_wgp = (gt_instances, ours_wgp_instances),
        ours_wsent = (gt_instances, ours_wsent_instances),
        ours_wsem = (gt_instances, ours_wsem_instances)
    )

    for key, values in metrics_dict.items():
        pred_euclidean_distance = list()
        pred_jawlips_distance = list()
        pred_eyebrows_distance = list()
        pred_mouth_distance = list()

        pred_landmarks_velocity = list()
        pred_jl_velocity = list()
        pred_eyeb_velocity = list()

        pred_mouth_opening = list()
        pred_eyeb_mdist = list()
        pred_mouth_mdist = list()
        pred_jl_mdist = list()
        pred_mdist = list()
        dtws = list()
        maes = list()
        for gt_instance, pred_instance in zip(values[0], values[1]):

            pred_mouth_opening.append(mouth_opening(pred_instance, gt_instance))
            pred_eyeb_mdist.append(get_eyebrows_mdiff(mean_face, pred_instance))
            pred_mouth_mdist.append(get_mouth_mdiff(mean_face, pred_instance))
            pred_jl_mdist.append(get_jl_mdiff(mean_face, pred_instance))
            pred_mdist.append(get_mdiff(mean_face, pred_instance))

            pred_euclidean_distance.append(get_euclidean_distance(pred_instance, gt_instance))
            pred_jawlips_distance.append(get_euclidean_distance(get_jaw_lips(pred_instance), get_jaw_lips(gt_instance)))
            pred_eyebrows_distance.append(get_euclidean_distance(get_eyebrows(pred_instance), get_eyebrows(gt_instance)))
            pred_mouth_distance.append(get_euclidean_distance(get_mouth(pred_instance), get_mouth(gt_instance)))

            pred_landmarks_velocity.append(get_landmarks_velocity(pred_instance, gt_instance))
            pred_jl_velocity.append(get_landmarks_velocity(get_jaw_lips(pred_instance), get_jaw_lips(gt_instance)))
            pred_eyeb_velocity.append(get_landmarks_velocity(get_eyebrows(pred_instance), get_eyebrows(gt_instance)))

            d, cost_matrix, acc_cost_matrix, path = dtw(pred_instance.reshape(-1, 136), gt_instance.reshape(-1, 136), dist = euclidean_norm)
            d = d/acc_cost_matrix.shape[0]
            dtws.append(d)
            maes.append(get_mae(pred_instance, gt_instance))

        """
        dtws = [x for x in dtws if x < 10]
        pred_jl_mdist = [x for x in pred_jl_mdist if x < 1]
        pred_mdist = [x for x in pred_mdist if x < 1]
        pred_mouth_opening = [x for x in pred_mouth_opening if x < 1000]
        pred_eyeb_mdist = [x for x in pred_eyeb_mdist if x < 1]
        pred_mouth_mdist = [x for x in pred_mouth_mdist if x < 1]
        pred_euclidean_distance = [x for x in pred_euclidean_distance if x < 1]
        pred_jawlips_distance = [x for x in pred_jawlips_distance if x < 1]
        pred_eyebrows_distance = [x for x in pred_eyebrows_distance if x < 1]
        pred_mouth_distance = [x for x in pred_mouth_distance if x < 1]
        pred_landmarks_velocity = [x for x in pred_landmarks_velocity if x < 1]
        pred_jl_velocity = [x for x in pred_jl_velocity if x < 1]
        pred_eyeb_velocity = [x for x in pred_eyeb_velocity if x < 1]
        """
        avg_mouth_opening = sum(pred_mouth_opening)/len(pred_mouth_opening)
        avg_eyeb_mdist = sum(pred_eyeb_mdist)/len(pred_eyeb_mdist)
        avg_mouth_mdist = sum(pred_mouth_mdist)/len(pred_mouth_mdist)
        avg_jl_mdist = sum(pred_jl_mdist)/len(pred_jl_mdist)
        avg_mdist = sum(pred_mdist)/len(pred_mdist)
        avg_euclidean_distance = sum(pred_euclidean_distance)/len(pred_euclidean_distance)
        avg_jawlips_distance = sum(pred_jawlips_distance)/len(pred_jawlips_distance)
        avg_eyebrows_distance = sum(pred_eyebrows_distance)/len(pred_eyebrows_distance)
        avg_mouth_distance = sum(pred_mouth_distance)/len(pred_mouth_distance)

        avg_landmarks_velocity = sum(pred_landmarks_velocity)/len(pred_landmarks_velocity)
        avg_jl_velocity = sum(pred_jl_velocity)/len(pred_jl_velocity)
        avg_eyeb_velocity = sum(pred_eyeb_velocity)/len(pred_eyeb_velocity)
        avg_dtw = sum(dtws)/len(dtws)
        avg_mae = sum(maes)/len(maes)

        if key == "ours":
            name = "Ours"
        elif key == "ours_wgp":
            name = "Ours w/o GP"
        elif key == "ours_wsem":
            name = "Ours w/o Sem"
        else:
            name = "Ours w/o Sent"

        df[name] = [
            avg_euclidean_distance,
            avg_jawlips_distance,
            avg_eyebrows_distance,
            avg_mouth_distance,
            avg_landmarks_velocity,
            avg_jl_velocity,
            avg_eyeb_velocity,
            avg_mouth_opening,
            avg_mouth_mdist,
            avg_eyeb_mdist,
            avg_jl_mdist,
            avg_mdist,
            avg_dtw,
            avg_mae
        ]

    df = pd.DataFrame.from_dict(df)
    df = df.round(4)
    df.to_csv("metrics_ablation.csv", sep = ";")

    """
    baseline_euclidean_distance = list()
    baseline_jawlips_distance = list()
    baseline_eyebrows_distance = list()
    baseline_mouth_distance = list()
    baseline_landmarks_velocity = list()
    baseline_jl_velocity = list()
    baseline_eyeb_velocity = list()

    ours_euclidean_distance = list()
    ours_jawlips_distance = list()
    ours_eyebrows_distance = list()
    ours_mouth_distance = list()
    ours_landmarks_velocity = list()
    ours_jl_velocity = list()
    ours_eyeb_velocity = list()

    ours_mouth_opening = list()
    baseline_mouth_opening = list()
    ours_eyeb_movement = list()
    baseline_eyeb_movement = list()
    ours_mouth_mdiff = list()

    for instance_name in os.listdir(gts_root):

        try:
            gt_instance = _read_instance_gt(gts_root, instance_name)
            baseline_instance = _read_makeittalk_instance(baseline_root, instance_name.replace(".pkl", ".npy"))
            #baseline_instance = _read_pt_instance(baseline_root, instance_name.replace(".pkl", ".npy"))[:,:-1].reshape(-1, 68, 2)
            ours_instance = _read_ours_instance(ours_root, instance_name.replace(".pkl", ".npz"))[:, :-1, :]
        except:
            print("found no existing instance, skipping")
            import traceback
            print(traceback.format_exc())
            continue
        
        baseline_kpsf = format_kps(baseline_instance)
        baseline_resampled_kps = np.asarray(sample_frames(baseline_kpsf)[1:])

        gt_kps = np.asarray(gt_instance["kps"])[:, :-1, :,]

       
        gt_kps = gt_kps*256
        baseline_resampled_kps = baseline_resampled_kps*256
        ours_instance = ours_instance*256
        

    
        ours_mouth_opening.append(mouth_opening(ours_instance, gt_kps))
        baseline_mouth_opening.append(mouth_opening(baseline_resampled_kps, gt_kps))

        ours_eyeb_movement.append(get_eyebrows_mdiff(mean_face, ours_instance))
        baseline_eyeb_movement.append(get_eyebrows_mdiff(mean_face, baseline_resampled_kps))

        ours_mouth_mdiff.append(get_mouth_mdiff(mean_face, ours_instance))

        baseline_euclidean_distance.append(get_euclidean_distance(baseline_resampled_kps, gt_kps))
        baseline_jawlips_distance.append(get_euclidean_distance(get_jaw_lips(baseline_resampled_kps), get_jaw_lips(gt_kps)))
        baseline_eyebrows_distance.append(get_euclidean_distance(get_eyebrows(baseline_resampled_kps), get_eyebrows(gt_kps)))
        baseline_mouth_distance.append(get_euclidean_distance(get_mouth(baseline_resampled_kps), get_mouth(gt_kps)))

        baseline_landmarks_velocity.append(get_landmarks_velocity(baseline_resampled_kps, gt_kps))
        baseline_jl_velocity.append(get_landmarks_velocity(get_jaw_lips(baseline_resampled_kps), get_jaw_lips(gt_kps)))
        baseline_eyeb_velocity.append(get_landmarks_velocity(get_eyebrows(baseline_resampled_kps), get_eyebrows(gt_kps)))

        ours_euclidean_distance.append(get_euclidean_distance(ours_instance, gt_kps))
        ours_jawlips_distance.append(get_euclidean_distance(get_jaw_lips(ours_instance), get_jaw_lips(gt_kps)))
        ours_eyebrows_distance.append(get_euclidean_distance(get_eyebrows(ours_instance), get_eyebrows(gt_kps)))
        ours_mouth_distance.append(get_euclidean_distance(get_mouth(ours_instance), get_mouth(gt_kps)))

        ours_landmarks_velocity.append(get_landmarks_velocity(ours_instance, gt_kps))
        ours_jl_velocity.append(get_landmarks_velocity(get_jaw_lips(ours_instance), get_jaw_lips(gt_kps)))
        ours_eyeb_velocity.append(get_landmarks_velocity(get_eyebrows(ours_instance), get_eyebrows(gt_kps)))


    ours_mouth_mdiff = [x for x in ours_mouth_mdiff if x < 1] 
    baseline_eyeb_movement = [x for x in baseline_eyeb_movement if x < 1] 
    ours_eyeb_movement = [x for x in ours_eyeb_movement if x < 1]
    ours_mouth_opening = [x for x in ours_mouth_opening if x < 1000]
    ours_euclidean_distance = [x for x in ours_euclidean_distance if x < 1]
    ours_jawlips_distance = [x for x in ours_jawlips_distance if x < 1]
    ours_eyebrows_distance = [x for x in ours_eyebrows_distance if x < 1]
    ours_mouth_distance = [x for x in ours_mouth_distance if x < 1]
    ours_landmarks_velocity = [x for x in ours_landmarks_velocity if x < 1]
    ours_jl_velocity = [x for x in ours_jl_velocity if x < 1]
    ours_eyeb_velocity = [x for x in ours_eyeb_velocity if x < 1]

    avg_ours_mdiff = sum(ours_mouth_mdiff)/len(ours_mouth_mdiff)
    #print("lele")
    #print(avg_ours_mdiff)


    avg_eyeb_movement = sum(ours_eyeb_movement)/len(ours_eyeb_movement)
    print("lele")
    print(avg_eyeb_movement)
    import pdb
    pdb.set_trace()
    avg_baseline_eyeb_movement = sum(baseline_eyeb_movement)/len(baseline_eyeb_movement)

    avg_mouth_opening = sum(ours_mouth_opening)/len(ours_mouth_opening)
    avg_baseline_mouth_opening = sum(baseline_mouth_opening)/len(baseline_mouth_opening)

    baseline_avg_dist = sum(baseline_euclidean_distance)/len(baseline_euclidean_distance)
    baseline_jl_dist = sum(baseline_jawlips_distance)/len(baseline_jawlips_distance)
    baseline_eyeb_dist = sum(baseline_eyebrows_distance)/len(baseline_eyebrows_distance)
    baseline_m_dist = sum(baseline_mouth_distance)/len(baseline_mouth_distance)

    baseline_avg_velocity_dist = sum(baseline_landmarks_velocity)/len(baseline_landmarks_velocity)
    baseline_jl_velocity_dist = sum(baseline_jl_velocity)/len(baseline_jl_velocity)
    baseline_eyeb_velocity_dist = sum(baseline_eyeb_velocity)/len(baseline_eyeb_velocity)

    ours_avg_dist = sum(ours_euclidean_distance)/len(ours_euclidean_distance)
    ours_jl_dist = sum(ours_jawlips_distance)/len(ours_jawlips_distance)
    ours_eyeb_dist = sum(ours_eyebrows_distance)/len(ours_eyebrows_distance)
    ours_m_dist = sum(ours_mouth_distance)/len(ours_mouth_distance)

    ours_avg_velocity_dist = sum(ours_landmarks_velocity)/len(ours_landmarks_velocity)
    ours_jl_velocity_dist = sum(ours_jl_velocity)/len(ours_jl_velocity)
    ours_eyeb_velocity_dist = sum(ours_eyeb_velocity)/len(ours_eyeb_velocity)

    df = dict(
        Ours = [ours_avg_dist, ours_jl_dist, ours_eyeb_dist, ours_m_dist, ours_avg_velocity_dist, ours_jl_velocity_dist, ours_eyeb_velocity_dist, avg_eyeb_movement],
        MakeItTalk = [baseline_avg_dist, baseline_jl_dist, baseline_eyeb_dist, baseline_m_dist, baseline_avg_velocity_dist, baseline_jl_velocity_dist, baseline_eyeb_velocity_dist, avg_baseline_eyeb_movement]
    )

    df = pd.DataFrame.from_dict(df)
    df.to_csv("metrics.csv", sep = ";")

    """

if __name__ == "__main__":
    main()