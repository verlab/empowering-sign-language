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
from models.navigator_net import Navigator



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
        fields = [('src', TEXT), ('kps', RAW), ("sent_feature", RAW), ("sem_feature", RAW), ("pos", RAW), ("aus", RAW), ("z", RAW), ("idx", RAW), ("file_name", RAW)]
    )

    iterator = data.BucketIterator(
        dataset = dataset,
        batch_size=batch_size,
        device=device, 
        sort_key=lambda x: len(x.src),
        repeat=False, 
        sort=False, 
        shuffle=True if train else False,
        sort_within_batch= True if train else False
    )

    return iterator

def save_ckpt(model, out_fp):
    torch.save(model.state_dict(), out_fp)

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

def validate(test_iterator, navigator, Z, device):
    epoch_loss = []
    navigator.eval()
    for idx, batch in enumerate(iter(test_iterator)):
        sent_feature = torch.Tensor(batch.sent_feature).to(device)
        sem_feature = torch.Tensor(batch.sem_feature).to(device)
        Zi = torch.Tensor(Z[batch.idx]).to(device).view(-1,768*8)        
        Z_prev = navigator(sent_feature,sem_feature)        
        loss = torch.sum(1 - fnn.cosine_similarity(Zi,Z_prev))
        epoch_loss.append(loss.item())
    print("Validation loss: {}".format(sum(epoch_loss)/len(epoch_loss)))

def main(args):

    SEED = 1234
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda:{}'.format(args.device)) if args.device != -1 else torch.device('cuda')
    N_EPOCHS = args.epochs
    experiment_name = "lr:{}-bs:{}-e:{}-ts:{}".format(
        args.learning_rate, 
        args.batch_size, 
        args.epochs,
        datetime.datetime.now()
    )

    experiment_folder = os.path.join(args.tensorboard_root, experiment_name)
    os.makedirs(experiment_folder, exist_ok = True)
    tb_writer = SummaryWriter(log_dir = experiment_folder)    
    
    dataset_root = args.dataset_root
    #encoder = SentiTransformerEncoder().to(device)
    decoder = Decoder(device).to(device)
    decoder.load_state_dict(torch.load(args.decoder_ckpt))
    decoder.eval()
    
    navigator = Navigator(768*2,768*8).to(device)
    
    train_iterator = get_dataset(
        args.dataset_root, 
        args.batch_size, 
        device, 
        train = True
    )

    test_iterator = get_dataset(
        args.dataset_root, 
        args.batch_size, 
        device, 
        train = False
    )
    
    optimizer_n = optim.AdamW([
        {'params': navigator.parameters(), 'lr': args.learning_rate, 'betas' : (0.5, 0.999)}
    ])
    
    scheduler_n = StepLR(optimizer_n, step_size = args.save_interval, gamma = 0.5)
    
    Z = read_pickle(args.zs_ckpt)

    mse = nn.MSELoss()
    
    step = 0
    
    for epoch in range(N_EPOCHS):        
        epoch_loss = []

        for idx, batch in enumerate(iter(train_iterator)):
            sent_feature = torch.Tensor(batch.sent_feature).to(device)
            sem_feature = torch.Tensor(batch.sem_feature).to(device)
            Zi = torch.Tensor(Z[batch.idx]).to(device).view(-1,768*8)
            
            optimizer_n.zero_grad()
            Z_prev = navigator(sent_feature,sem_feature)
            
            cosine_distance = torch.sum(1 - fnn.cosine_similarity(Zi,Z_prev))

            mse_distance = mse(Zi, Z_prev)
           
            loss = cosine_distance
            loss.backward()
            optimizer_n.step()
            scheduler_n.step()

            tb_writer.add_scalar("train/Cosine Distance Loss", loss.item(), step)
            tb_writer.add_scalar("train/L2 Loss", mse_distance.item(), step)
            epoch_loss.append(loss.item())
            
            step = step + 1 
            
            if step % args.save_interval == 0 and step > 0:
                validate(test_iterator, navigator, Z, device)
                navigator.train()
                try:
                    model_folder = os.path.join(args.ckpt_root, experiment_name, "step_{}".format(step))
                    os.makedirs(model_folder, exist_ok = True)

                    network_fp = os.path.join(model_folder, "navigator.pth")
                    save_ckpt(navigator, network_fp)

                except:
                    import traceback
                    print(traceback.format_exc())
                    exit()
                    #encoder.train()            
            
            
        print("Epoch loss: {} -- Step: {}".format(sum(epoch_loss)/len(epoch_loss), step))  


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument('--device', '-dev', type = int, default = 0)
    parser.add_argument('--batch_size', '-bs', type = int, default = 42)
    parser.add_argument('--epochs', '-e', type = int, default = 1000)     
    parser.add_argument('--save_interval', '-si', type = int, default = 12500)
    parser.add_argument('--dataset_root', '-d', type = str, default = "/srv/storage/datasets/rafaelvieira/slp_dataset_f68_full_ps_new_arch_sample_wz_final_grammar_wsem/train")
    parser.add_argument('--validation_root', '-dv', type = str, default = "/srv/storage/datasets/rafaelvieira/slp_dataset_f68_full_ps_new_arch_sample_wz_final_grammar_wsem/test")
    parser.add_argument('--outputs_root', '-ot', type = str, default = "/srv/storage/datasets/rafaelvieira/text2expression/results/")
    parser.add_argument('--decoder_ckpt', '-dckpt', type = str, default = "/srv/storage/datasets/rafaelvieira/text2expression/checkpoints_new_arch_debug/lrdec:0.015-lrdisc:0.015-lrsenc:0.015-lridenc:0.015-bs:24-e:200000-ts:2022-10-05 20:08:06.699304/step_137500/decoder.pth")
    parser.add_argument('--zs_ckpt', '-zckpt', type = str, default = "/srv/storage/datasets/rafaelvieira/text2expression/checkpoints_new_arch_debug/lrdec:0.015-lrdisc:0.015-lrsenc:0.015-lridenc:0.015-bs:24-e:200000-ts:2022-10-05 20:08:06.699304/step_137500/Zs.pkl")
   
    parser.add_argument('--learning_rate', '-lr', type = float, default = 0.0001)
    parser.add_argument('--learning_rate_idenc', '-lridenc', type = float, default = 0.015)
    parser.add_argument('--learning_rate_dec', '-lrdec', type = float, default = 0.015)
    parser.add_argument('--learning_rate_disc', '-lrdisc', type = float, default = 0.015)

 
    parser.add_argument('--ckpt_root', '-ct', type = str, default = "/srv/storage/datasets/rafaelvieira/SenteRetri_small_wtanh_adamw2/models/")
    parser.add_argument('--tensorboard_root', '-tr', type = str, default = "/srv/storage/datasets/rafaelvieira/SenteRetri_small_wtanh_adamw2/tensorboard/")


    args = parser.parse_args()
    main(args)
