# -*- coding: utf-8 -*-
#for some reason this import has to come first, otherwise gives segfault
from email import iterators
import sys
sys.path.append("..")

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
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from utils.util import Utils
from torchtext import data
from torchtext import datasets
from torch.utils.data import Dataset, DataLoader
from models.pose_autoencoder import EmbeddingNet
#from models.decoder import Decoder
from data.dataset import SignProdDataset, SignProdDatasetPhoenix
from sentence_transformers import SentenceTransformer
import pdb

tokenizer = BertTokenizer.from_pretrained('/srv/storage/datasets/rafaelvieira/transformersw/', local_files_only = True)
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

def save_ckpt(model, out_fp):
    torch.save(model.state_dict(), out_fp)

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

  
    train_iterator = SignProdDatasetPhoenix(args.dataset_root)
    test_iterator = SignProdDatasetPhoenix(args.validation_root)

    train_iterator = DataLoader(train_iterator, batch_size=128, shuffle=True, num_workers=4)
    test_iterator = DataLoader(test_iterator, batch_size=128, shuffle=True, num_workers=4)
    
    print ("Finished load data")
   
    # interval params
    save_model_epoch_interval = args.interval

    # init model and optimizer

    # pose_dim = 68*2,n_frames
    generator = EmbeddingNet(240, 64).to(device)
    gen_optimizer = optim.Adam(generator.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
    scheduler_gen = StepLR(gen_optimizer, step_size = 15000, gamma = 0.5)

    step = 0

    step_test = 0


    global_iter = 0
    best_values = {}  # best values for all loss metrics
    for epoch in range(args.epochs):

        print("Epoch, step: " + str(epoch) + "," + str(step))

        generator.train(False) 

        with torch.no_grad():
 
            for idx, kps in enumerate(iter(test_iterator)):

                #get data

                kps = kps.to(device)
                feat, a, b, recon_poses = generator(kps)

                #compute loss

                recon_loss = F.l1_loss(recon_poses, kps, reduction='none')
                recon_loss = torch.mean(recon_loss, dim=(1, 2))
                target_diff = kps[:, 1:] - kps[:, :-1]        
                recon_diff = recon_poses[:, 1:] - recon_poses[:, :-1]        
                recon_loss += torch.mean(F.l1_loss(recon_diff, target_diff, reduction='none'), dim=(1, 2))
                recon_loss = torch.sum(recon_loss)
                loss = recon_loss    
                print("Validation loss : {}".format(loss.item()))     
            
                tb_writer.add_scalar("test/Loss", loss.item(), step_test)

                step_test += 1

        if (epoch % save_model_epoch_interval == 0 and epoch > 0):

            if True:
                try:
                    model_folder = os.path.join(args.ckpt_root, experiment_name, "step_{}".format(step))
                    os.makedirs(model_folder, exist_ok = True)
                    autoencoder = os.path.join(model_folder, "autoencoder.pth")
                    save_ckpt(generator,autoencoder)
                    print("Save model")

                except:
                    import pdb
                    pdb.set_trace()


        # train iter

        generator.train(True) 

        for idx, kps in enumerate(iter(train_iterator)):

            #get data
            kps = kps.to(device)
            gen_optimizer.zero_grad()

            feat, a, b, recon_poses = generator(kps)

            #compute loss

            recon_loss = F.l1_loss(recon_poses, kps, reduction='none')
            recon_loss = torch.mean(recon_loss, dim=(1, 2))
            target_diff = kps[:, 1:] - kps[:, :-1]        
            recon_diff = recon_poses[:, 1:] - recon_poses[:, :-1]        
            recon_loss += torch.mean(F.l1_loss(recon_diff, target_diff, reduction='none'), dim=(1, 2))
            recon_loss = torch.sum(recon_loss)
            loss = recon_loss
         
            if step%10 == 0:
                print (loss.item())

            loss.backward()
            gen_optimizer.step()
            scheduler_gen.step()

            # print training status

            tb_writer.add_scalar("train/Loss", loss.item(), step)

            step += 1

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument('--learning_rate', '-lr', type = float, default = 0.0005)
    parser.add_argument('--device', '-dev', type = int, default = 0)
    parser.add_argument('--batch_size', '-bs', type = int, default = 24)
    parser.add_argument('--epochs', '-e', type = int, default = 500)
    parser.add_argument('--interval', '-int', type = int, default = 40)
    parser.add_argument('--dataset_root', '-d', type = str, default =  "/srv/storage/datasets/thiagocoutinho/datasets/phoenix/train")
    parser.add_argument('--validation_root', '-dv', type = str, default =  "/srv/storage/datasets/thiagocoutinho/datasets/phoenix/dev")
    parser.add_argument('--ckpt_root', '-ct', type = str, default = "/srv/storage/datasets/rafaelvieira/checkpoints_fgd_bs_128_scheduler_long")
    parser.add_argument('--outputs_val_root', '-ov', type = str, default = "/srv/storage/datasets/rafaelvieira/checkpoints_fgd_bs_128_scheduler_long/")
    parser.add_argument('--outputs_train_root', '-ot', type = str, default = "/srv/storage/datasets/rafaelvieira/checkpoints_fgd_bs_128_scheduler_long/")
    parser.add_argument('--tensorboard_root', '-tr', type = str, default = "/srv/storage/datasets/rafaelvieira/checkpoints_fgd_bs_128_scheduler_long/tensorboard")

    args = parser.parse_args()
    main(args)
