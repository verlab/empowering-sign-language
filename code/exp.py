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
        test_input = torch.Tensor(test_input).unsqueeze(0).to(device)
        faces_fake = decoder(None, None, test_input, device).permute(0, 2, 3, 1).cpu().numpy()
        Utils.visualize_data_single(faces_fake, exp_name)


def main():

    decoder_ckpt_fp = "/srv/storage/datasets/rafaelvieira/text2expression/checkpoints_new_arch_overfitting4_wz_glo_adam_equallr_Z_all_lap/decoder.pth"
    zs_ckpt_fp = "/srv/storage/datasets/rafaelvieira/text2expression/checkpoints_new_arch_overfitting4_wz_glo_adam_equallr_Z_all_lap/Zs.pkl"
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda:0")

    decoder = Decoder(device).to(device)
    decoder.load_state_dict(torch.load(decoder_ckpt_fp))
    Z = read_pickle(zs_ckpt_fp)
    #encoder.train()
    decoder.eval()

    with torch.no_grad():
        #test_input = Z[130] + Z[20]
        test_input1 = Z[91] - Z[2]
        test_input4 = Z[91] + Z[2]
        test_input2 = Z[91]
        test_input3 = Z[2]
        make_exp(test_input1, "91-2.mp4", decoder, Z, device)
        make_exp(test_input4, "91+2.mp4", decoder, Z, device)
        make_exp(test_input2, "91.mp4", decoder, Z, device)
        make_exp(test_input3, "2.mp4", decoder, Z, device)

if __name__ == "__main__":
    main()