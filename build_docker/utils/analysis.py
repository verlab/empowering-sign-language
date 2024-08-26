from transformers import BertTokenizer, BertModel
import re
import os
import sys
import pickle
import copy
import cv2
import uuid
import numpy as np
import torch
import math
import shutil
from sklearn.preprocessing import StandardScaler
from frontalization.utils import frontalize_landmarks
from sklearn.decomposition import PCA
from euro_filter import OneEuroFilter
from scipy.stats import multivariate_normal
from util import Utils
from sentence_transformers import SentenceTransformer
#from audio_utils import AudioUtil
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
max_input_length = tokenizer.max_model_input_sizes['bert-base-uncased']
frontalization_weights = np.load('frontalization/data/frontalization_weights.npy')
N_frames = 64

def read_data(file_name):
    with open(file_name, "rb") as handler:
        return pickle.load(handler)

def read_data_npz(file_name):
    info = np.load(file_name, allow_pickle=True)
    dataset = info['dataset'][()]
    return dataset

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

def draw_keypoints(kps):
    img = np.ones((256, 256, 3))*255

    for kp in kps:
        x = kp[0]
        y = kp[1]
        img = cv2.circle(img, (int(x*256), int(y*256)), 2, (0, 255 ,0), 2)
    return img

def filter_jittering(x_noisy):

    min_cutoff = 0.05
    beta = 0.7
    x_hat = np.zeros_like(x_noisy)
    x_hat[0] = x_noisy[0]
    t = np.arange(0, len(x_noisy))

    one_euro_filter = OneEuroFilter(
        t[0], x_noisy[0],
        min_cutoff=min_cutoff,
        beta=beta
    )

    for i in range(1, len(t)):
        x_hat[i] = one_euro_filter(t[i], x_noisy[i])

    return x_hat

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
    for idx in range(0, 69):
        formatted_kps[:,idx,0] = filter_jittering(formatted_kps[:,idx,0])
        formatted_kps[:,idx,1] = filter_jittering(formatted_kps[:,idx,1])

    return formatted_kps.tolist()

def visualize_data(predicted, gt, out_root):
    uuid_s = uuid.uuid4()
    out_folder = os.path.join(out_root, str(uuid_s))
    os.makedirs(out_folder, exist_ok = True)
    out_fp = os.path.join(out_folder, "{}.mp4".format(str(uuid.uuid4())))
    predicted_imgs = [Utils.draw_keypoints(kp) for kp in predicted]
    gts_imgs = [Utils.draw_keypoints(kp) for kp in gt]
    Utils.plot_video(predicted_imgs, gts_imgs, out_fp)

def remove_kps(kps, idxs = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 19, 20 ,21, 22, 23, 24, 25, 28, 30, 32, 33, 34, 48, 50, 62, 52, 54, 56, 66, 58, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]):
    kps_lst = kps.tolist()
    new_lst = list()

    for idx, kp in enumerate(kps_lst):
        if idx in idxs:
            new_lst.append(kp)

    return np.asarray(new_lst)

def upsample_tokens(sentences, N_TOKENS = 128):
    upsampled_tokens = list()
    for tokens in sentences:
        idx_t = 0
        idxs = np.linspace(0, len(tokens), num = N_TOKENS)[:-1]
        for idx in idxs:
            upsampled_tokens.append(tokens[int(idx)])
        upsampled_tokens.append(tokens[int(idxs[-1])])
    return upsampled_tokens

def kps_knn(kps, k = 4):
    edges = list()

    for idx, kp in enumerate(kps):
        edges_kp = get_kp_knn(kp, kps, idx, k = k)
        edges.append(edges_kp)

    return edges

def get_kp_knn(kp, kps, kp_idx, k = 5):
    distances = list()

    for idx, kp_n in enumerate(kps):
        if kp_idx == idx:
            continue
        dist = np.linalg.norm(kp - kp_n)
        distances.append((idx, dist))

    sorted_dist = sorted(distances, key = lambda x: x[1], reverse = False)[:k]
    edges = [(kp_idx, x[0]) for x in sorted_dist]
    return edges

def flatten(t):
    return [item for sublist in t for item in sublist]

def project_l2_ball(z):
    """ project the vectors in z onto the l2 unit norm ball"""
    return z / np.maximum(np.sqrt(np.sum(z**2, axis=1))[:, np.newaxis], 1)

def generate_gp(idx, c=512, m=4):
    np.random.seed(idx)
    xs = np.linspace(0,1000,m) # Test input vector
    mxs = np.zeros(m) # Zero mean vector

    z = []
    for i in range(c):
        lsc = ((float(i)+1)/c)*(100*(1024/c))
        Kss = np.exp((-1*(xs[:,np.newaxis]-xs[:,np.newaxis ].T)**2)/(2*lsc**2)) # Covariance matrix
        fs = multivariate_normal(mean=mxs ,cov=Kss , allow_singular =True).rvs(1).T
        z.append(fs)
    z = np.asarray(z)
    return z

def _read_instance(instance_fp):
    with open(instance_fp, "rb") as handler:
        instance = pickle.load(handler)
    return instance

def _read_instance_gt(instance_fp):
    with open(instance_fp, "rb") as handler:
        instance = pickle.load(handler)
    return instance

def get_original_features(features):
    previous = None
    original_features = list()

    for idx in range(0, features.shape[0]):
        features_lst = features[idx].tolist()
        if previous == None or previous != features_lst:
            previous = features[idx].tolist()
            original_features.append(features[idx])
    


    if len(original_features) < 137:
        diff = 137 - len(original_features)
        for idx in range(0, diff):
            original_features.append(np.zeros((768,)).tolist())

    original_features = np.asarray(original_features)
    return original_features

def generate_fake_tokens(size_):
    return [10 if idx < size_ else -10000 for idx in range(0, 137)]

def get_pos(sentence_name, df):
    row = df.loc[df["videos_names"] == sentence_name]
    pos = row["pos"].tolist()[0]
    pos = pos.strip('][').split(', ')
    pos = [p.replace("'", "") for p in pos]
    return pos

def get_aus(file_fp):
    #aus_names = [' AU04_r', ' AU05_r', ' AU17_r', ' AU45_r', 'AU1415', 'AU0102', 'AU2526']
    with open(file_fp, "rb") as handler:
        data = pickle.load(handler)
    return data

def get_sent_embeddings(file_fp):
    with open(file_fp, "rb") as handler:
        data = pickle.load(handler)
    return data["sentence_embeddings"]

def sample_aus(aus):
    all_sampled_aus = list()
    for key, value in aus.items():
        aus = list()
        for idx in sorted(value.keys()):
            aus.append(value[idx])
        sampled_aus = sample_frames(aus)
        all_sampled_aus.append(sampled_aus)
    all_sampled_aus = np.asarray(all_sampled_aus)

    reframed_aus = list()
    for idx in range(0, all_sampled_aus[0].shape[0]):
        reframed_aus.append(all_sampled_aus[:, idx].tolist())
    return reframed_aus

def pad_sentence(tokens, max_seq_len):
    diff = max_seq_len - len(tokens)
    for _ in range(0, diff):
        tokens.append(0)
    return tokens

def build_pos_dict():
    return dict(
        ADJ = 1,
        ADP = 2,
        ADV = 3,
        AUX = 4,
        CONJ = 5,
        CCONJ = 6,
        DET = 7,
        INTJ = 8,
        NOUN = 9,
        NUM = 10,
        PART = 11,
        PRON = 12,
        PROPN = 13,
        PUNCT = 14,
        SCONJ = 15,
        SYM = 16,
        VERB = 17,
        X = 18,
        SPACE = 19
    )


def permute_pos(pos, pos_map, n = 8):
    import random
    tries = 0

    for try_ in range(0, 5000):
        print("Trying {}".format(try_))
        idxs = random.sample(range(0, len(pos)), n)
        pos_selected = [p for idx, p in enumerate(pos) if idx in idxs]

        if ",".join(pos_selected) not in pos_map:
            pos_map.add(",".join(pos_selected))
            return pos_selected, pos_map
    return None, pos_map

def generate_pos_tokens():
    pos_dict = build_pos_dict()
    gp_dict = dict()
    for idx, key in enumerate(pos_dict.keys()):
        key_gp = generate_gp(78979 + idx, c = 768, m = 1)
        gp_dict[key] = key_gp
    return gp_dict 

def clean_kps(kps):
    mean_face = np.load("mean_face_h2s_sp1.npy")/256
    cleaned_kps = list()
    
    for kp in kps:
        kp = np.asarray(kp)
        dist = np.linalg.norm(kp - mean_face)
        print(dist)
        if dist <= 0.6:
            cleaned_kps.append(kp)
    
    print("Before cleaning: {}".format(len(kps)))
    print("After cleaning: {}".format(len(cleaned_kps)))
    return cleaned_kps

def get_speakers_num(df, dataset):
    map_ = dict()
    for key, value in dataset.items():
        speaker_id = key[14:]
        speaker_id = re.search(r'\d+', speaker_id).group()

        if speaker_id not in map_:
            map_[speaker_id] = 1
        else:
            map_[speaker_id] += 1
    return map_

def main():

    device = torch.device("cuda:0")
    #bert = BertModel.from_pretrained('bert-base-uncased').to(device)
    #bert.eval()
    s_bert = SentenceTransformer('sentence-transformers/all-distilroberta-v1').to(device)
    s_bert.eval()

    if sys.argv[1] == "merge":

        old_dataset_fp = "/home/rafaelv/masters/datasets/slp_dataset_f68_new2/slp_dataset_f68/train/"
        full_dataset_fp = "/home/rafaelv/masters/datasets/slp_dataset_f68_full_ps_test_bigger_new/slp_dataset_f68_full_ps_test_bigger/train/"
        merged_dataset_fp = "/home/rafaelv/masters/datasets/slp_dataset_f68_merged/slp_dataset_f68/train/"

        old_files = os.listdir(old_dataset_fp)
        full_files = os.listdir(full_dataset_fp)

        idx = 2500
        for file_full in full_files:
            if file_full not in old_files:
                gp = generate_gp(idx, m = 8, c = 768)
                instance_fp = os.path.join(full_dataset_fp, file_full)
                merged_file_fp = os.path.join(merged_dataset_fp, file_full)
                instance = _read_instance_gt(instance_fp)
                instance["z"] = gp
                with open(merged_file_fp, "wb") as handler:
                    pickle.dump(instance, handler)
                print("Wrote idx {}".format(idx))
                idx += 1

    if sys.argv[1] == "generate_new":
        import pandas as pd
        type_ = sys.argv[2]
        s_bert = SentenceTransformer('sentence-transformers/all-distilroberta-v1').to(device)
        s_bert.eval()   
        dataset_root = "/home/rafael/data/how2sign_dataset_speaker1_clean/"
        emb_root = "/home/rafael/data/new_sent_embeddings_test"
        dataset_root = os.path.join(dataset_root, type_)
        h2s_root = "/home/rafael/data/how2sign/test_sample.npz"
        os.makedirs(dataset_root, exist_ok = True)
        dataset = read_data_npz(h2s_root)
        idx = 0
        max_seq_len = 33
        min_seq_len = 10
        errors = 0

        #signer01 and signer05 are the ones with most samples
        #signer04 and signer08 are the new ones
        df_root = "/home/rafael/data/how2sign_realigned_test.csv"
        df = pd.read_csv(df_root, sep = "|")
        #df = df.loc[df["speaker"] == "Signer04"]
        map_ = get_speakers_num(df, dataset)
        #speaker_instances = df["name"].tolist()
        for key, value in dataset.items():
            speaker_id = key[14:]
            speaker_id = re.search(r'\d+', speaker_id).group()

            if speaker_id != "1":
                continue

            print("Running idx: {}".format(idx))

            kps = dataset[key]['kp']
            kps = np.asarray(kps)
            text = dataset[key]['text']
            splitted_text = text.split(" ")
            
            if len(kps) <= 192:
                instance_fp = os.path.join(dataset_root, "{}.pkl".format(key))
                sent_emb_fp = os.path.join(emb_root, "{}.pkl".format(key))
                embeddings = get_sent_embeddings(sent_emb_fp)

                kps = kps[:, :, :-1]
                kps = format_kps(kps)
                kps = clean_kps(kps)

                if len(kps) >= 45:
                    resampled_kps = sample_frames(kps)
                    gp = generate_gp(idx, m = 8, c = 768)

                    with torch.no_grad():
                        s_feature = s_bert.encode(text, convert_to_tensor = True)
                    s_feature = s_feature.cpu().numpy()

                    data = dict(
                        kps = resampled_kps,
                        text = text,
                        sent_embeddings = embeddings,
                        sem_embeddings = s_feature,
                        z = gp,
                    )
                    print("Writing data for: {}".format(instance_fp))
                    with open(instance_fp, "wb") as handler:
                        pickle.dump(data, handler)
                    idx += 1

    if sys.argv[1] == "generate":
        import pandas as pd
        type_ = sys.argv[2]
        dataset_root = "/srv/storage/datasets/rafaelvieira/slp_dataset_f68_full_ps_new_arch_sample_wz_final"
        pos_root = "/srv/storage/datasets/rafaelvieira/new_data/{}_pos.csv".format(type_)
        aus_root = "/srv/storage/datasets/rafaelvieira/new_data/aus_selected"
        emb_root = "/srv/storage/datasets/rafaelvieira/new_data/new_sent_embeddings"
        pos_df = pd.read_csv(pos_root, sep = ";")
        dataset_root = os.path.join(dataset_root, type_)
        h2s_root = "/srv/storage/datasets/rafaelvieira/how2sign/train_sample.npz"
        os.makedirs(dataset_root, exist_ok = True)    
        dataset = read_data(h2s_root)
        idx = 0
        max_seq_len = 33
        min_seq_len = 10
        pos_dict = build_pos_dict()
        errors = 0
        for value, key in dataset.items():
            try:
                speaker_id = value[14:]
                speaker_id = re.search(r'\d+', speaker_id).group()
            except:
                import traceback
                print(traceback.format_exc())
                continue
            if speaker_id == "8":
                print("Running idx: {}".format(idx))
                kps = dataset[value]['kp']
                text = dataset[value]['text']
                splitted_text = text.split(" ")
                try:
                    if len(kps) <= 192 and len(splitted_text) >= min_seq_len and len(splitted_text) <= max_seq_len:
                        instance_fp = os.path.join(dataset_root, "{}.pkl".format(value))
                        aus_fp = os.path.join(aus_root, "{}_postprocessed.pkl".format(value))
                        sent_emb_fp = os.path.join(emb_root, "{}.pkl".format(value))
                        pos = get_pos(value, pos_df)
                        aus = get_aus(aus_fp)
                        aus = sample_aus(aus)
                        embeddings = get_sent_embeddings(sent_emb_fp)
                        kps = format_kps(kps[:, :, :-1])
                        resampled_kps = sample_frames(kps)
                        pos = [pos_dict[p] for p in pos]
                        gp = generate_gp(idx, m = 8, c = 768)

                        data = dict(
                            kps = resampled_kps,
                            text = text,
                            sent_embeddings = embeddings,
                            aus = aus,
                            pos = pos,
                            z = gp
                        )
                        print("Writing data for: {}".format(instance_fp))
                        with open(instance_fp, "wb") as handler:
                            pickle.dump(data, handler)
                        idx += 1
                except:
                    errors += 1
                    print("Total errors: {}".format(errors))

                    #mel_spec = process_audio(os.path.join(audio_root, "{}.wav".format(value)))
                    """
                    kps = format_kps(kps)
                    resampled_kps = sample_frames(kps)

                    if resampled_kps:
                        assert len(resampled_kps) == N_frames
                        marked_text = "[CLS] " + text + " [SEP]"
                        tokenized_text = tokenizer.tokenize(marked_text)
                        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
                        tokens_tensor = torch.tensor([indexed_tokens]).to(device)

                        with torch.no_grad():
                            t_feature = bert(tokens_tensor)[0]
                            s_feature = s_bert.encode(text, convert_to_tensor = True)

                        s_feature = s_feature.cpu().numpy()
                        t_feature = np.asarray(upsample_tokens(t_feature.cpu().numpy().tolist()))
                        instance_fp = os.path.join(dataset_root, "{}.pkl".format(value))
                        print("Writing data for: {}".format(instance_fp))
                        id_kp = resampled_kps[0]
                        resampled_kps = resampled_kps[1:]
                        gp = generate_gp(idx, m = 8, c = 768)
                        data = dict(
                            kps = resampled_kps, 
                            text = text, 
                            id_kp = id_kp, 
                            t_feature = t_feature,
                            s_feature = s_feature,
                            #mel_spec = mel_spec, 
                            z = gp
                        )

                        with open(instance_fp, "wb") as handler:
                            pickle.dump(data, handler)
                        idx += 1
                    """
    if sys.argv[1] == "edges":
        template_lmk = np.load("frontalization/data/landmarks_mean_face.npy")
        #template_lmk = remove_kps(template_lmk)
        edges = kps_knn(template_lmk, k = 3)
        edges = flatten(edges)
        edges = [np.array(e) for e in edges]
        edges = np.array(edges)
        np.save("edges.npy", edges)
        lala = np.load("edges.npy")

    if sys.argv[1] == "showtext":
        val_root = "/home/rafaelv/masters/datasets/slp_dataset_f68_new2/slp_dataset_f68/val/"
        files = os.listdir(val_root)
        texts = list()
        names = list()
        for file in files:
            file_fp = os.path.join(val_root, file)
            instance = _read_instance(file_fp)
            text = instance["text"]
            texts.append(text)
            names.append(file)

        df = pd.DataFrame.from_dict(dict(texts = texts, names = names))
        df.to_csv("texts.csv", sep = ";")
        import pdb
        pdb.set_trace()

    if sys.argv[1] == "count":
        h2s_root = "/home/rafaelv/masters/datasets/how2sign/train_sample.npz"
        dataset = read_data(h2s_root)
        idx = 0
        max_seq_len = 96
        count = 0

        speaker_dict = dict()
        for value, key in dataset.items():
            try:
                speaker_id = value[14:]
                speaker_id = re.search(r'\d+', speaker_id).group()
            except:
                import traceback
                print(traceback.format_exc())
                continue

            if speaker_id not in speaker_dict:
                speaker_dict[speaker_id] = 0
            else:
                current_count = speaker_dict[speaker_id]
                current_count += 1
                speaker_dict[speaker_id] = current_count


        import pdb
        pdb.set_trace()
        print("Total: {}".format(count))


    if sys.argv[1] == "gen-tsv":
        data_root = "/home/rafaelv/masters/datasets/slp_dataset_f68_full_ps5_new/slp_dataset_f68_full_ps5/val/"    
        #dataset = read_data(h2s_root)
        output_root = "/home/rafaelv/masters/datasets/audio_data_ps5/"
        os.makedirs(output_root, exist_ok = True)
        texts = list()
        outputs = list()
        mocked_mels = list()
        mel_name = "mels/test.pt"
        idx = 0
        files = os.listdir(data_root)

        for file_ in files:
            file_fp = os.path.join(data_root, file_)
            instance = _read_instance(file_fp)
            text = instance["text"]
            output_file = "{}.wav".format(file_.replace(".pkl", ""))
            texts.append(text)
            outputs.append(output_file)
            mocked_mels.append(mel_name)

        df = dict(mel = mocked_mels, output = outputs, text = texts)
        df = pd.DataFrame.from_dict(df)
        df.to_csv("test_ps5.tsv", sep = "\t", index = False)

    if sys.argv[1] == "vocab":
        dataset_root = "/home/rafaelv/masters/datasets/slp_dataset_f68/"
        h2s_root = "/home/rafaelv/masters/datasets/how2sign/train_sample.npz"    
        dataset = read_data(h2s_root)
        vocab = set()
        for value, key in dataset.items():
            text = dataset[value]["text"]
            text_splitted = text.split(" ")

            for t in text_splitted:
                t = t.encode("utf-8").strip()
                if t not in vocab:
                    vocab.add(t)

        handler = open("src_vocab.txt", "a")
        for t in vocab:
            decoded = t.decode().replace(",", "").replace("!", "").replace("?", "").replace(".", "")
            try:
                handler.write(decoded.upper())
                handler.write("\n")
            except:
                pass
        handler.close()

    if sys.argv[1] == "mean":
        phases = ["train"]
        dataset_root = "/home/rafael/data/how2sign_dataset_speaker1/"
        kps_cumm = list()
        total_len = 0
        for phase in phases:
            files_ = os.listdir(os.path.join(dataset_root, phase))
            for idx, file_ in enumerate(files_):
                print("Processing instance: {}".format(idx))
                file_fp = os.path.join(os.path.join(dataset_root, phase), file_)
                instance = _read_instance(file_fp)
                kps = np.array(instance["kps"])*256
                kps = kps.sum(axis = 0)
                kps_cumm.append(kps)
                total_len += 64

        kps_cumm = np.asarray(kps_cumm)
        kps_mean = kps_cumm.sum(axis = 0)/total_len
        np.save("mean_face_h2s_sp1.npy", kps_mean)

    if sys.argv[1] == "std":
        phases = ["train", "val"]
        dataset_root = "/srv/storage/datasets/rafaelvieira/slp_dataset_f68_merged_c/slp_dataset_f68"
        mean_face_fp = "mean_face.npy"
        mean_face = np.load(mean_face_fp)
        kps_std = list()
        total_len = 0
        for phase in phases:
            files_ = os.listdir(os.path.join(dataset_root, phase))
            for idx, file_ in enumerate(files_):
                print("Processing instance: {}".format(idx))
                file_fp = os.path.join(os.path.join(dataset_root, phase), file_)
                instance = _read_instance(file_fp)
                kps = np.array(instance["kps"])*256
                kps_diff = np.power(kps - mean_face, 2)
                kps_sum = kps_diff.sum(axis = 0)
                kps_std.append(kps_sum)
                total_len += 64
        

        kps_std = np.asarray(kps_std)
        kps_std = np.sqrt(kps_std.sum(axis = 0)/total_len)
        np.save("std_face.npy", kps_std)

    if sys.argv[1] == "norm":
        phases = ["train", "val"]
        dataset_root = "/srv/storage/datasets/rafaelvieira/slp_dataset_f68_merged_c/slp_dataset_f68"
        std_face = np.load("std_face.npy")
        mean_face = np.load("mean_face.npy")
        kps_std = list()
        total_len = 0
        for phase in phases:
            files_ = os.listdir(os.path.join(dataset_root, phase))
            for idx, file_ in enumerate(files_):
                print("Processing instance: {}".format(idx))
                file_fp = os.path.join(os.path.join(dataset_root, phase), file_)
                instance = _read_instance(file_fp)
                
                kps = np.array(instance["kps"])*256
                kps_norm = (kps - mean_face)

                instance["kps_norm"] = kps_norm

                with open(file_fp, "wb") as handler:
                    pickle.dump(instance, handler)

    if sys.argv[1] == "clean":
        import shutil
        dataset_root = "/home/rafael/data/phoenix_dataset_signer04/test"
        files = os.listdir(dataset_root)
        mean_face = np.load("mean_face_phoenix_sp4.npy")/256
        files_names = list()
        
        for file_ in files:
            print("Veryfing instance {}".format(file_))
            file_fp = os.path.join(dataset_root, file_)
            instance = _read_instance_gt(file_fp)
            kps = instance["kps"]
            
            for kp in kps:
                kp = np.asarray(kp)
                dist = np.linalg.norm(kp - mean_face)
                
                if dist > 0.85 and file_ not in files_names:
                    img = draw_keypoints(kp)
                    #cv2.imwrite("{}.jpg".format(file_), img)
                    files_names.append(file_)
               
        for file_ in files_names:
            shutil.move(os.path.join(dataset_root, file_), os.path.join("/home/rafael/buggy_instances_train_sp4", file_))

    if sys.argv[1] == "reorg":
        dataset_root = "/srv/storage/datasets/rafaelvieira/slp_dataset_f68_merged_c/slp_dataset_f68"
        type_ = sys.argv[2]
        dataset_root = os.path.join(dataset_root, type_)
        files = os.listdir(dataset_root)

        for idx, file_ in enumerate(files):
            print("Processing instance: {}".format(idx))
            file_fp = os.path.join(dataset_root, file_)
            instance = _read_instance_gt(file_fp)
            text = instance["text"]
            marked_text = "[CLS] " + text + " [SEP]"
            tokenized_text = tokenizer.tokenize(marked_text)
            indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
            tokens_tensor = torch.tensor([indexed_tokens]).to(device)

            with torch.no_grad():
                t_feature = bert(tokens_tensor)[0]

            original_t = get_original_features(instance["t_feature"])
            original_ts = get_original_features(instance["ts_feature"])

            instance["original_t"] = original_t
            instance["original_ts"] = original_ts

            with open(file_fp, "wb") as handler:
                pickle.dump(instance, handler)

    if sys.argv[1] == "pad":
        dataset_root = "/srv/storage/datasets/rafaelvieira/slp_dataset_f68_merged_c/slp_dataset_f68"
        type_ = sys.argv[2]
        dataset_root = os.path.join(dataset_root, type_)
        files = os.listdir(dataset_root)

        for idx, file_ in enumerate(files):
            print("Processing instance: {}".format(idx))
            file_fp = os.path.join(dataset_root, file_)
            instance = _read_instance_gt(file_fp)
            text = instance["text"]
            marked_text = "[CLS] " + text + " [SEP]"
            tokenized_text = tokenizer.tokenize(marked_text)
            indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
            tokens_tensor = torch.tensor([indexed_tokens]).to(device)
            fake_tokens = generate_fake_tokens(tokens_tensor.shape[-1])
            instance["fake_token"] = fake_tokens

            with open(file_fp, "wb") as handler:
                pickle.dump(instance, handler)

    if sys.argv[1] == "counttokens":
        import matplotlib.pyplot as plt
    
        dataset_root = "/srv/storage/datasets/rafaelvieira/slp_dataset_f68_merged_c/slp_dataset_f68"
        type_ = sys.argv[2]
        dataset_root = os.path.join(dataset_root, type_)
        files = os.listdir(dataset_root)
        sizes_dict = dict()

        for idx, file_ in enumerate(files):
            print("Processing instance: {}".format(idx))
            file_fp = os.path.join(dataset_root, file_)
            instance = _read_instance_gt(file_fp)
            text = instance["text"]
            splitted_text = text.split(" ")
            len_ = len(splitted_text)

            if len_ in sizes_dict:
                sizes_dict[len_] += 1
            else:
                sizes_dict[len_] = 1
        
        import pdb
        pdb.set_trace()

    if sys.argv[1] == "remove":

        type_ = sys.argv[2]
        dataset_root = "/srv/storage/datasets/rafaelvieira/slp_dataset_f68_merged_c/slp_dataset_f68/"
        output_root = "/srv/storage/datasets/rafaelvieira/slp_dataset_f68_merged_c_sampled/"
        dataset_root = os.path.join(dataset_root, type_)
        output_root = os.path.join(output_root, type_)
        h2s_root = "/srv/storage/datasets/rafaelvieira/how2sign/train_sample.npz"
        os.makedirs(dataset_root, exist_ok = True)
        os.makedirs(output_root, exist_ok = True)    
        dataset = read_data(h2s_root)
        import shutil
        idx = 0
        for value, key in dataset.items():
            try:
                speaker_id = value[14:]
                speaker_id = re.search(r'\d+', speaker_id).group()
            except:
                import traceback
                print(traceback.format_exc())
                continue
            if speaker_id == "8":
                print("Running idx: {}".format(idx))
                kps = dataset[value]['kp']
                instance_fp = os.path.join(dataset_root, "{}.pkl".format(value))
                if len(kps) <= 192 and os.path.isfile(instance_fp):
                    output_fp = os.path.join(output_root, "{}.pkl".format(value))

                    print("Moving {} to {}".format(instance_fp, output_fp))
                    shutil.copy(instance_fp, output_fp)
                    idx += 1


        
        

    





if __name__ == "__main__":
    main()
