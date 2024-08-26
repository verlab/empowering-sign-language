import os
import sys
import pickle
import numpy as np
import random
import torch
import torchvision
from utils.util import Utils
from models.decoder_stgcn import Decoder
from data.dataset import SignProdDataset

def read_pickle(file_fp):
    with open(file_fp, "rb") as handler:
        return pickle.load(handler)

def make_exp(test_input, exp_name, decoder, device):
    with torch.no_grad():
        test_input = torch.Tensor(test_input).unsqueeze(0).to(device)
        import pdb
        pdb.set_trace()
        faces_fake = decoder(None, None, test_input).permute(0, 2, 3, 1).cpu().numpy()
        Utils.visualize_data_single(faces_fake, exp_name)

def printinfo(dataset_root, file):
    file_fp = os.path.join(dataset_root, file)
    instance = read_pickle(file_fp)
    print("Instance label: {}".format(instance["label"]))
    print("Instance name: {}".format(file))


def filter_speaker(files):
    filtered_files = list()
    import re
    for file_ in files:
        speaker_id = file_[14:]
        speaker_id = re.search(r'\d+', speaker_id).group()
        if speaker_id == "8":
            filtered_files.append(file_)
    return filtered_files


def read_embeddings(sent_embeddings_root):
    embeddings = dict()
    files = os.listdir(sent_embeddings_root)
    files = filter_speaker(files)
    for file_ in files:
        print(file_)
        file_fp = os.path.join(sent_embeddings_root, file_)
        instance = read_pickle(file_fp)
        embeddings[file_] = instance
    return embeddings

def filter_embeddings(sent_embeddings, label):
    new_dict = dict()
    for file_, embeddings in sent_embeddings.items():
        if embeddings["label"] == label:
            new_dict[file_] = embeddings
    return new_dict

def get_instance_idx(instance_name, files):
    for idx, file_ in enumerate(files):
        if instance_name == file_:
            return idx

def inference(Z, Zs, decoder, device):

    zi = torch.zeros((1, 768, 8))
    zi = torch.autograd.Variable(zi, requires_grad=False)

    zs = torch.zeros((1, 768, 8))
    zs = torch.autograd.Variable(zs, requires_grad=False)

    with torch.no_grad():
        #test_input = Z[130] + Z[20]
        zi.data = torch.Tensor(Z)
        zs.data = torch.Tensor(Zs)
        input_ = zi + zs
        input_ = input_.unsqueeze(0).to(device)
        faces_fake = decoder(None, None, input_).permute(0, 2, 3, 1).cpu().numpy()
        
    return faces_fake

def is_not_mean(mean_face, kps):
    for kp in kps[0]:
        kp = np.asarray(kp)
        dist = np.linalg.norm(kp - mean_face)
        if dist > 0.17:
            return True
    return False

def main():
    
    dataset_root = "/srv/storage/datasets/rafaelvieira/slp_dataset_f68_full_ps_new_arch_sample_wz_final_grammar/train"
    sent_root = "/srv/storage/datasets/rafaelvieira/new_data/new_sent_embeddings"
    decoder_ckpt_fp = "/srv/storage/datasets/rafaelvieira/text2expression/should_be_good/decoder.pth"
    zs_ckpt_fp = "/srv/storage/datasets/rafaelvieira/text2expression/should_be_good/Zs.pkl"
    zsent_ckpt_fp = "/srv/storage/datasets/rafaelvieira/text2expression/should_be_good/Zsent.pkl"
    output_root = "/srv/storage/datasets/rafaelvieira/text2expression/sentiment_results_preliminar"
    mean_face = np.load("mean_face.npy")/256

    os.makedirs(output_root, exist_ok=True)
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda:0")

    decoder = Decoder(device).to(device)
    decoder.load_state_dict(torch.load(decoder_ckpt_fp))

    Z = read_pickle(zs_ckpt_fp)
    Zs = read_pickle(zsent_ckpt_fp)
    
    decoder.eval()
    files = sorted(os.listdir(dataset_root))[:6071]
    print("Reading embeddings")
    sent_embeddings = read_embeddings(sent_root)
    labels_names = ["anger", "fear", "joy", "sadness"]

    for idx, file_ in enumerate(files):

        print("Reading file {}".format(file_))
        sent_file_fp = os.path.join(sent_root, file_)
        file_fp = os.path.join(dataset_root, file_)

        if not os.path.isfile(sent_file_fp):
            continue
        
        if file_ in sent_embeddings:
            ori_label = sent_embeddings[file_]["label"]
            for label in labels_names:
                label_embeddings = filter_embeddings(sent_embeddings, label)
                len_ = len(label_embeddings)
                import random
                population = len_ if len_ < 3 else 3
                sent_idxs = random.sample(range(len_), population)
                #test_input_normal = Z[idx] + Zs[idx]
                faces_fake_normal = inference(Z[idx], Zs[idx], decoder, device)

                for video_idx, s_idx in enumerate(sent_idxs):
                    instance_idx = None

                    while instance_idx is None:
                        instance_name = list(label_embeddings.keys())[s_idx]
                        instance_idx = get_instance_idx(instance_name, files)

                        if instance_idx is None:
                            s_idx = int(random.random()*len_)

                    #test_input_switch = Z[idx] + Zs[instance_idx]
                    faces_fake_switch = inference(Z[idx] , Zs[instance_idx], decoder, device)

                    if is_not_mean(mean_face, faces_fake_switch):

                        switch_name = "{}_ori={}_swi={}_vid={}.npz".format(file_.replace(".pkl", ""), ori_label, label, video_idx)
                        normal_name = "{}_normal.npz".format(file_.replace(".pkl", ""))

                        video_switch_name = "{}_ori={}_swi={}_vid={}.mp4".format(file_.replace(".pkl", ""), ori_label, label, video_idx)
                        video_normal_name = "{}_normal.mp4".format(file_.replace(".pkl", ""))

                        output_folder = os.path.join(output_root, file_.replace(".pkl", ""))
                        os.makedirs(output_folder, exist_ok = True)
                        switch_fp = os.path.join(output_folder, switch_name)
                        normal_fp = os.path.join(output_folder, normal_name)
                        video_switch_fp = os.path.join(output_folder, video_switch_name)
                        video_normal_fp = os.path.join(output_folder, video_normal_name)

                        np.savez(switch_fp, faces_fake_switch.squeeze(0))
                        np.savez(normal_fp, faces_fake_normal.squeeze(0))

                        Utils.visualize_data_single(faces_fake_normal, video_normal_fp)
                        Utils.visualize_data_single(faces_fake_switch, video_switch_fp)

if __name__ == "__main__":
    main()
