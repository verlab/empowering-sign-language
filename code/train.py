# -*- coding: utf-8 -*-
#for some reason this import has to come first, otherwise gives segfault
from transformers import BertTokenizer, BertModel
import datetime
import argparse
import uuid
import torch
import random
import numpy as np
import pickle
import os
import cv2
import copy
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
from utils.util import Utils
from torchtext import data
from torchtext import datasets
from torch.utils.data import Dataset, DataLoader
from models.decoder_stgcn import Decoder
from models.sent_transformers import SentiTransformerEncoder
from data.dataset import SignProdDataset

from torch import nn
import torch.nn.functional as fnn
from torch.autograd import Variable
from torch.optim import SGD
from torchvision.datasets import LSUN
from torchvision import transforms
from torch.utils.data import Dataset
from torchvision.utils import make_grid

PAD_IDX = 0
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
max_input_length = tokenizer.max_model_input_sizes['bert-base-uncased']

def tokenize_and_cut(sentence):
    tokens = tokenizer.tokenize(sentence) 
    tokens = tokens[:max_input_length-2]
    return tokens

def upsample_tokens(sentences, N_TOKENS = 128):
    upsampled = list()

    for tokens in sentences:
        idxs = np.linspace(0, len(tokens) - 1, num = (N_TOKENS - len(tokens)))
        tokens_ = copy.deepcopy(tokens)
        for idx in idxs:
            tokens_.insert(int(idx), tokens[int(idx)])
        upsampled.append(tokens_)
    return upsampled

def get_dataset(data_root, batch_size, device, train = True):

    init_token_idx = tokenizer.convert_tokens_to_ids(tokenizer.cls_token)
    eos_token_idx = tokenizer.convert_tokens_to_ids(tokenizer.sep_token)
    pad_token_idx = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    unk_token_idx = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)

    TEXT = data.Field(
        batch_first = True,
        use_vocab = False,
        tokenize = tokenize_and_cut,
        preprocessing = tokenizer.convert_tokens_to_ids,
        init_token = init_token_idx,
        eos_token = eos_token_idx,
        pad_token = pad_token_idx,
        unk_token = unk_token_idx
    )

    RAW = data.RawField()

    dataset = SignProdDataset(
        data_root,
        train,
        fields = [('src', TEXT), ('kps', RAW), ("s_feature", RAW), ("pos", RAW), ("aus", RAW), ("z", RAW), ("idx", RAW), ("file_name", RAW)]
    )

    iterator = data.BucketIterator(
        dataset = dataset,
        batch_size=batch_size,
        device=device, 
        sort_key=lambda x: len(x.src),
        repeat=False, 
        sort=False, 
        shuffle=True if train else False,
        sort_within_batch=True
    )

    return iterator

def validate(encoder, decoder, loss, iterator, device, Z, Zs, zi, zs, step, unit_tensor):
    all_losses = list()
    file_names = list()

    #encoder.eval()
    decoder.eval()

    for idx, batch in enumerate(iter(iterator)):

        kps = torch.Tensor(np.asarray(batch.kps)).to(device).permute(0, 3, 1, 2)
        s_feature = torch.Tensor(batch.s_feature).to(device)
        d_idxs = np.asarray(batch.idx)

        with torch.no_grad():

            zi.data = (torch.Tensor(Z[d_idxs]).to(device))
            faces_fake = decoder(None, s_feature, zi)
            loss_t = loss(faces_fake, kps)

            if idx == 0:
                gen_faces = copy.deepcopy(faces_fake)
                kps_t = copy.deepcopy(kps)
                file_names = batch.file_name
            else:
                gen_faces = torch.cat([gen_faces, faces_fake], dim = 0)
                kps_t = torch.cat([kps_t, kps], dim = 0)
                file_names += batch.file_name

        all_losses.append(loss_t.item())
    return sum(all_losses)/len(all_losses), gen_faces, kps_t, file_names

def save_ckpt(model, out_fp):
    torch.save(model.state_dict(), out_fp)

def create_mask(src):
    src_seq_len = src.shape[0]
    src_mask = torch.zeros((src_seq_len, src_seq_len))
    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    return src_mask, src_padding_mask

def project_l2_ball(z):
    """ project the vectors in z onto the l2 unit norm ball"""
    return z / np.maximum(np.sqrt(np.sum(z**2, axis=1))[:, np.newaxis], 1)

def project_l2_ball_torch(zt, unit_tensor):
    return zt/torch.maximum(torch.sqrt(torch.sum(zt**2, axis = 1))[:, None], unit_tensor)

def build_gauss_kernel(size=5, sigma=1.0, n_channels=1, cuda=False):
    if size % 2 != 1:
        raise ValueError("kernel size must be uneven")
    grid = np.float32(np.mgrid[0:size,0:size].T)
    gaussian = lambda x: np.exp((x - size//2)**2/(-2*sigma**2))**2
    kernel = np.sum(gaussian(grid), axis=2)
    kernel /= np.sum(kernel)
    # repeat same kernel across depth dimension
    kernel = np.tile(kernel, (n_channels, 1, 1))
    # conv weight should be (out_channels, groups/in_channels, h, w), 
    # and since we have depth-separable convolution we want the groups dimension to be 1
    kernel = torch.FloatTensor(kernel[:, None, :, :])
    if cuda:
        kernel = kernel.cuda()
    return Variable(kernel, requires_grad=False)


def conv_gauss(img, kernel):
    """ convolve img with a gaussian kernel that has been built with build_gauss_kernel """
    n_channels, _, kw, kh = kernel.shape
    img = fnn.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
    return fnn.conv2d(img, kernel, groups=n_channels)


def laplacian_pyramid(img, kernel, max_levels=5):
    current = img
    pyr = []

    for level in range(max_levels):
        filtered = conv_gauss(current, kernel)
        diff = current - filtered
        pyr.append(diff)
        current = fnn.avg_pool2d(filtered, 2)

    pyr.append(current)
    return pyr


class LapLoss(nn.Module):
    def __init__(self, max_levels=5, k_size=5, sigma=2.0):
        super(LapLoss, self).__init__()
        self.max_levels = max_levels
        self.k_size = k_size
        self.sigma = sigma
        self._gauss_kernel = None
        
    def forward(self, input, target):
        if self._gauss_kernel is None or self._gauss_kernel.shape[1] != input.shape[1]:
            self._gauss_kernel = build_gauss_kernel(
                size=self.k_size, sigma=self.sigma, 
                n_channels=input.shape[1], cuda=input.is_cuda
            )
        pyr_input  = laplacian_pyramid( input, self._gauss_kernel, self.max_levels)
        pyr_target = laplacian_pyramid(target, self._gauss_kernel, self.max_levels)
        return sum(fnn.l1_loss(a, b) for a, b in zip(pyr_input, pyr_target))

def _read_instance(instance_fp):
    with open(instance_fp, "rb") as handler:
        instance = pickle.load(handler)
    return instance

def read_zs(dataset_root):
    Z = list()
    files = os.listdir(dataset_root)
    for file_ in sorted(files):
        file_fp = os.path.join(dataset_root, file_)
        instance = _read_instance(file_fp)
        emb = instance["z"][:, :2]
        Z.append(emb)
    return np.asarray(Z)


def read_sents(dataset_root):
    Z = list()
    files = os.listdir(dataset_root)
    for file_ in sorted(files):
        file_fp = os.path.join(dataset_root, file_)
        instance = _read_instance(file_fp)
        emb = instance["sent_embeddings"]
        emb_exp = np.expand_dims(emb, axis = -1)
        emb_exp = emb_exp.repeat(8, axis = -1)
        Z.append(emb_exp)
    return np.asarray(Z)

def save_Zs(output_root, Z):
    with open(output_root, "wb") as handler:
        pickle.dump(Z, handler)

def main(args):

    SEED = 1234
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda:{}'.format(args.device)) if args.device != -1 else torch.device('cuda')
    N_EPOCHS = args.epochs
    experiment_name = "lrdec:{}-lrdisc:{}-lrsenc:{}-lridenc:{}-bs:{}-e:{}-ts:{}".format(
        args.learning_rate_dec, 
        args.learning_rate_disc,
        args.learning_rate_senc, 
        args.learning_rate_idenc, 
        args.batch_size, 
        args.epochs,
        datetime.datetime.now()
    )

    experiment_folder = os.path.join(args.tensorboard_root, experiment_name)
    os.makedirs(experiment_folder, exist_ok = True)

    #encoder = SentiTransformerEncoder().to(device)
    decoder = Decoder(device).to(device)

    #encoder.train()
    decoder.train()

    zi = torch.zeros((args.batch_size, 768, 2))
    zi = Variable(zi, requires_grad=True)

    optimizer_g = optim.Adam([
        {'params': decoder.parameters(), 'lr': args.learning_rate_dec, 'betas' : (0.5, 0.999)},
        {'params': zi, 'lr': args.learning_rate_dec}
    ])

    scheduler_g = StepLR(optimizer_g, step_size = 15000, gamma = 0.5)

    lap_loss = LapLoss(max_levels=3)
    #lap_loss = nn.L1Loss().to(device)
 
    step = 0

    Z = read_zs(args.dataset_root)
    Z = project_l2_ball(Z)
    
    train_iterator = get_dataset(
        args.dataset_root, 
        args.batch_size, 
        device, 
        train = True
    )

    test_iterator = get_dataset(
        args.validation_root, 
        args.batch_size, 
        device, 
        train = False
    )

    unit_tensor = torch.Tensor([1]).to(device)
    for epoch in range(N_EPOCHS):
        
        epoch_loss = []

        for idx, batch in enumerate(iter(train_iterator)):
            optimizer_g.zero_grad()

            kps = torch.Tensor(np.asarray(batch.kps)).to(device).permute(0, 3, 1, 2)
            s_feature = torch.Tensor(batch.s_feature).to(device)
            d_idxs = np.asarray(batch.idx)
            zi.data = torch.Tensor(Z[d_idxs]).to(device)
        
            faces_fake = decoder(None, s_feature, zi)

            loss = lap_loss(faces_fake, kps)

            loss.backward()
            optimizer_g.step()
            scheduler_g.step()

            Z[d_idxs] = project_l2_ball(zi.data.cpu().numpy())
            
            epoch_loss.append(loss.item())

            if step % 12500 == 0 and step > 0:
                try:

                    model_folder = os.path.join(args.ckpt_root, experiment_name, "step_{}".format(step))
                    os.makedirs(model_folder, exist_ok = True)

                    decoder_fp = os.path.join(model_folder, "decoder.pth")
                    dict_fp = os.path.join(model_folder, "Zs.pkl")

                    save_ckpt(decoder, decoder_fp)
                    save_Zs(dict_fp, Z)

                    val_loss, faces_fake_val, kps_val, file_names = validate(None, decoder, lap_loss, test_iterator, device, Z, None, zi, None, step, unit_tensor)
                    Utils.visualize_data(faces_fake_val, kps_val, args.outputs_val_root, step, file_names)

                    print("Current validation loss: {}".format(val_loss))
                    decoder.train()
                    #encoder.train()
                except:
                    import traceback
                    print(traceback.format_exc())
                    decoder.train()
                    #encoder.train()
            
            step += 1
        
        print("Epoch loss: {} -- Step: {}".format(sum(epoch_loss)/len(epoch_loss), step))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument('--learning_rate_senc', '-lrsenc', type = float, default = 0.015)
    parser.add_argument('--learning_rate_idenc', '-lridenc', type = float, default = 0.015)
    parser.add_argument('--learning_rate_dec', '-lrdec', type = float, default = 0.015)
    parser.add_argument('--learning_rate_disc', '-lrdisc', type = float, default = 0.015)
    parser.add_argument('--flip', '-f', type = float, default = 0.4)
    parser.add_argument('--device', '-dev', type = int, default = 0)
    parser.add_argument('--warm_up', '-wup', type = int, default = 6000)
    parser.add_argument('--batch_size', '-bs', type = int, default = 24)
    parser.add_argument('--epochs', '-e', type = int, default = 200000)
    parser.add_argument('--interval', '-int', type = int, default = 40)
    parser.add_argument('--dataset_root', '-d', type = str, default = "/home/rafael/data/how2sign_dataset_speaker1_clean/train/")
    parser.add_argument('--validation_root', '-dv', type = str, default = "/home/rafael/data/how2sign_dataset_speaker1_clean/train/")
    parser.add_argument('--ckpt_root', '-ct', type = str, default = "/home/rafael/masters/to_move/checkpoints_new_arch_modgcn_sp1h2s_clean/")
    parser.add_argument('--outputs_val_root', '-ov', type = str, default = "/home/rafael/masters/to_move/checkpoints_new_arch_modgcn_sp1h2s_clean/")
    parser.add_argument('--outputs_train_root', '-ot', type = str, default = "/home/rafael/masters/to_move/checkpoints_new_arch_modgcn_sp1h2s_clean/")
    parser.add_argument('--tensorboard_root', '-tr', type = str, default = "/home/rafael/masters/to_move/tensorboard_slp2/")
    #wconv2 has lap level = 3
    #wconv3 has lap level = 2
    args = parser.parse_args()
    main(args)
