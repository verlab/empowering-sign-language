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
from torch.utils.tensorboard import SummaryWriter
from utils.util import Utils
from torchtext import data
from torchtext import datasets
from torch.utils.data import Dataset, DataLoader
from models.decoder_stgcn import Decoder
from models.sent_transformers import SentiTransformerEncoder
from data.dataset import SignProdDataset
import spacy
import nltk
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
nlp = spacy.load("en_core_web_sm")

def tokenize_and_cut(sentence):
    tokens = tokenizer.tokenize(sentence) 
    tokens = tokens[:max_input_length-2]
    return tokens

def get_pos(text):
    doc = nlp(text)
    pos_list = list()
    for token in doc:
        pos_list.append(token.pos_)
    return pos_list

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

def save_Zs(output_root, Z):
    with open(output_root, "wb") as handler:
        pickle.dump(Z, handler)

def read_pickle(file_fp):
    with open(file_fp, "rb") as handler:
        return pickle.load(handler)

def read_sentences(dataset_root, files):
    sentences = list()
    for idx, file_ in enumerate(files[:50]):
        file_fp = os.path.join(dataset_root, file_)
        instance = read_pickle(file_fp)
        sentences.append((idx, instance["text"]))
    return sentences


def get_k_similar_sentences(pos, sentences, k):
    sentences_dist = dict()
    pos_rev_dict = build_pos_rev_dict()
    for idx, sentence in sentences:
        pos_s = get_pos(sentence)
        pos_s_mapped = [pos_rev_dict[p] for p in pos_s]
        pos_mapped = [pos_rev_dict[p] for p in pos]
        pos_s_mapped = "".join(pos_s_mapped)
        pos_mapped = "".join(pos_mapped)
        distance = nltk.edit_distance(pos_s_mapped, pos_mapped)

        if distance not in sentences_dist:
            sentences_dist[distance] = [idx]
        else:
            sentences_dist[distance].append(idx)
    
    sorted_keys = list(sorted(sentences_dist.keys()))
    selected_sentences = list()

    for key in sorted_keys:
        sentences = sentences_dist[key]
        for s in sentences:
            if len(selected_sentences) == k:
                return selected_sentences
            selected_sentences.append(s)
    
                
def build_pos_rev_dict():

    return dict(
        ADJ = "a",
        ADP = "b",
        ADV = "c",
        AUX = "d",
        CONJ = "e",
        CCONJ = "f",
        DET = "g",
        INTJ = "h",
        NOUN = "i",
        NUM = "j",
        PART = "k",
        PRON = "l",
        PROPN = "m",
        PUNCT = "n",
        SCONJ = "o",
        SYM = "p",
        VERB = "q",
        X = "r",
        SPACE = "s"
    )


def project_l2_ball(z):
    """ project the vectors in z onto the l2 unit norm ball"""
    return z / np.maximum(np.sqrt(np.sum(z**2, axis=1))[:, np.newaxis], 1)

def save_predictions(output_fp, faces_fake):
    np.savez(output_fp, faces_fake)


def main(args):

    SEED = 1234
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda:{}'.format(args.device)) if args.device != -1 else torch.device('cuda')
    N_EPOCHS = args.epochs
    dataset_root = args.dataset_root
    output_root = args.outputs_root
    os.makedirs(output_root, exist_ok = True)
    #encoder = SentiTransformerEncoder().to(device)
    decoder = Decoder(device).to(device)
    decoder.load_state_dict(torch.load(args.decoder_ckpt))
    decoder.eval()

    print("Reading Zs")
    Z = read_pickle(args.zs_ckpt)
    print("Reading Zsent")
    Zs = read_pickle(args.zsent_ckpt)

    files = sorted(os.listdir(dataset_root))
    print("Reading sentences")
    sentences = read_sentences(dataset_root, files)

    for file_ in files:
        file_fp = os.path.join(dataset_root, file_)
        output_fp = os.path.join(output_root, file_.replace(".pkl", ".npz"))
        instance = read_pickle(file_fp)
        kps = instance["kps"]
        text = instance["text"]

        pos = get_pos(text)
        selected_sentences = get_k_similar_sentences(pos, sentences, 2)
        z = 0 
        zs = 0

        for idx_s in selected_sentences:
            z += Z[idx_s]
            zs += Zs[idx_s]
        
        z = np.expand_dims(z, axis = 0)
        zs = np.expand_dims(zs, axis = 0)

        z = project_l2_ball(z)
        zs = project_l2_ball(zs)

        with torch.no_grad():
            zs = torch.Tensor(zs).to(device)
            z = torch.Tensor(z).to(device)
            input_ = z + zs
            faces_fake = decoder(None, None, input_).permute(0, 2, 3, 1).cpu().numpy()
        save_predictions(output_fp, faces_fake)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument('--device', '-dev', type = int, default = 0)
    parser.add_argument('--batch_size', '-bs', type = int, default = 42)
    parser.add_argument('--epochs', '-e', type = int, default = 200000)
    parser.add_argument('--dataset_root', '-d', type = str, default = "/srv/storage/datasets/rafaelvieira/slp_dataset_f68_full_ps_new_arch_sample_wz_final_grammar/test")
    parser.add_argument('--outputs_root', '-ot', type = str, default = "/srv/storage/datasets/rafaelvieira/text2expression/ours_results/")
    parser.add_argument('--decoder_ckpt', '-dckpt', type = str, default = "/srv/storage/datasets/rafaelvieira/text2expression/should_be_good/decoder.pth")
    parser.add_argument('--zs_ckpt', '-zckpt', type = str, default = "/srv/storage/datasets/rafaelvieira/text2expression/should_be_good/Zs.pkl")
    parser.add_argument('--zsent_ckpt', '-zsentckpt', type = str, default = "/srv/storage/datasets/rafaelvieira/text2expression/should_be_good/Zsent.pkl")

    args = parser.parse_args()
    main(args)
