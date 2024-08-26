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

def make_exp(test_input, exp_name, decoder, Z, device):
    with torch.no_grad():
        test_input = torch.Tensor(test_input).to(device)
        faces_fake = decoder(None, None, test_input).permute(0, 2, 3, 1).cpu().numpy()
        Utils.visualize_data_single(faces_fake, exp_name)
    return faces_fake

def printinfo(dataset_root, file):
    file_fp = os.path.join(dataset_root, file)
    instance = read_pickle(file_fp)
    print("Instance label: {}".format(instance["label"]))
    print("Instance name: {}".format(file))

def project_l2_ball(z):
    """ project the vectors in z onto the l2 unit norm ball"""
    return z / np.maximum(np.sqrt(np.sum(z**2, axis=1))[:, np.newaxis], 1)

def project_l2_ball_torch(zt, unit_tensor):
    return zt/torch.maximum(torch.sqrt(torch.sum(zt**2, axis = 1))[:, None], unit_tensor)

def is_not_mean(mean_face, kps):
    for kp in kps[0]:
        kp = np.asarray(kp)
        dist = np.linalg.norm(kp - mean_face)
        print(dist)
        if dist > 0.65:
            import pdb
            pdb.set_trace()
            return True
    return False


def main():
    dataset_root = "/srv/storage/datasets/rafaelvieira/new_data/new_sent_embeddings"
    decoder_ckpt_fp = "/srv/storage/datasets/rafaelvieira/text2expression/should_be_good/decoder.pth"
    zs_ckpt_fp = "/srv/storage/datasets/rafaelvieira/text2expression/should_be_good/Zs.pkl"
    zsent_ckpt_fp = "/srv/storage/datasets/rafaelvieira/text2expression/should_be_good/Zsent.pkl"
    mean_face = np.load("mean_face.npy")/256

    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda:0")
    unit_tensor = torch.Tensor([1])
    decoder = Decoder(device).to(device)
    decoder.load_state_dict(torch.load(decoder_ckpt_fp))
    Z = read_pickle(zs_ckpt_fp)
    Zs = read_pickle(zsent_ckpt_fp)
    #encoder.train()
    decoder.eval()
    files = sorted(os.listdir(dataset_root))

    idx1 = 2
    idx2 = 2

    printinfo(dataset_root, files[idx1])
    printinfo(dataset_root, files[idx2])

    zi = torch.zeros((1, 768, 8))
    zi = torch.autograd.Variable(zi, requires_grad=False)

    zs = torch.zeros((1, 768, 8))
    zs = torch.autograd.Variable(zs, requires_grad=False)

    with torch.no_grad():
        #test_input = Z[130] + Z[20]
        zi.data = (torch.Tensor(Z[idx1]))
        zs.data = (torch.Tensor(Zs[idx2]))
        input_ = zi + zs
        input_ = input_.unsqueeze(0)
        #input_ = project_l2_ball_torch(input_, unit_tensor)
        faces_fake = make_exp(input_, "youdont.mp4", decoder, Z, device)
        is_not_mean(mean_face, faces_fake)
        import pdb
        pdb.set_trace()
        print("heheh")

    """
    printinfo(dataset_root, files[idx1])
    printinfo(dataset_root, files[idx2])

    for idx, file_ in enumerate(files):
        print("Looking in video")
        file_fp = os.path.join(dataset_root, file_)
        instance = read_pickle(file_fp)
        label = instance["label"]

        if label != "joy":
            with torch.no_grad():
                #test_input = Z[130] + Z[20]
                test_input1 = Z[idx1] + Zs[idx1]
                test_input1 = project_l2_ball(test_input1)
                make_exp(test_input1, "video_{}.mp4".format(idx), decoder, Z, device)
    """        

if __name__ == "__main__":
    main()