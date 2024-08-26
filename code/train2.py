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
from torch.utils.data import DataLoader
from data.dataset_sn import CustomImageDataset

from torch import nn
import torch.nn.functional as fnn
from torch.autograd import Variable
from torch.optim import SGD
from torchvision.datasets import LSUN
from torchvision import transforms
from torch.utils.data import Dataset
from torchvision.utils import make_grid

def read_pickle(file_fp):
    with open(file_fp, "rb") as handler:
        return pickle.load(handler)

def project_l2_ball_torch(zt):
    unit_tensor = torch.Tensor([1])
    return zt/torch.maximum(torch.sqrt(torch.sum(zt**2, axis = 1))[:, None], unit_tensor)

class Net(nn.Module):

    def __init__(
        self, 
        in_channels, 
        out_channels, 
        depth=1,
        device = None
      ):
        super(Net, self).__init__()
        
        self.model = nn.Sequential(
          torch.nn.Linear(in_channels, out_channels), 
          torch.nn.Linear(out_channels, 2*out_channels), 
          torch.nn.Tanh(),
          torch.nn.Linear(2*out_channels, 2*out_channels),
          torch.nn.Tanh(),
          torch.nn.Linear(2*out_channels, out_channels),
          torch.nn.Tanh()        
        )
       
    def forward(self, sent_feature, sem_feature):
        x = torch.cat([sent_feature, sem_feature], dim = 1)
        x = self.model(x)
        return x

def validate(net, val_dataloader, loss_f, device):
    net.eval()
    epoch_loss = list()
    for sent_embedding, sem_embedding, z, zs in val_dataloader:
        embedding = embedding.permute(0, 2, 1).to(device)
        z = z.to(device)
        zs = zs.to(device)
        output = net(embedding)
        loss = loss_f(output, z + zs, torch.Tensor([1]).to(device))
        epoch_loss.append(loss.item())
    epoch_mean_loss = sum(epoch_loss)/len(epoch_loss)
    print("Validation Epoch Loss: {}".format(epoch_mean_loss))

def save_ckpt(model, out_fp):
    torch.save(model.state_dict(), out_fp)

def main(args):

    SEED = 1234
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda:0')

    Z = read_pickle(args.zs_ckpt)
    Zsent = read_pickle(args.zsent_ckpt)

    training_dataset = CustomImageDataset(args.dataset_root, Z, Zsent, train = True)
    validation_dataset = CustomImageDataset(args.dataset_root, Z, Zsent, train = False) 

    train_dataloader = DataLoader(training_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(validation_dataset, batch_size=32, shuffle=False)

    net = Net(768*2, 768*8).to(device)
    net.train()

    loss_f = torch.nn.MSELoss()

    optimizer_g = optim.Adam([
        {'params': net.parameters(), 'lr': args.learning_rate, 'betas' : (0.5, 0.999)}
    ])

    scheduler_g = StepLR(optimizer_g, step_size = 1000, gamma = 0.5)
    step = 0
    
    for epoch in range(0, 100):
        epoch_loss = list()
        for sent_embedding, sem_embedding, z, zs in train_dataloader:

            z = z.to(device)
            zs = zs.to(device)
            sent_embedding = sent_embedding.to(device)
            sem_embedding = sem_embedding.to(device)

            output = net(sent_embedding, sem_embedding).reshape(-1, 768, 8)
   
            loss = loss_f(output, z + zs)
            epoch_loss.append(loss.item())
            step += 1

            if step % 3000 == 0:
                validate(net, val_dataloader, loss_f, device)
                net.train()

            loss.backward()
            optimizer_g.step()
            scheduler_g.step()
        
        epoch_mean_loss = sum(epoch_loss)/len(epoch_loss)
        print("Train Epoch Loss: {}".format(epoch_mean_loss))
        
    save_ckpt(net, "/srv/storage/datasets/rafaelvieira/text2expression/proj.pth")
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument('--learning_rate', '-lr', type = float, default = 0.001)
    parser.add_argument('--device', '-dev', type = int, default = 0)
    parser.add_argument('--batch_size', '-bs', type = int, default = 42)
    parser.add_argument('--epochs', '-e', type = int, default = 200000)
    parser.add_argument('--dataset_root', '-d', type = str, default = "/srv/storage/datasets/rafaelvieira/slp_dataset_f68_full_ps_new_arch_sample_wz_final_grammar_wsem/train")
    parser.add_argument('--validation_root', '-dv', type = str, default = "/srv/storage/datasets/rafaelvieira/slp_dataset_f68_full_ps_new_arch_sample_wz_final_grammar_wsem/test")
    parser.add_argument('--zs_ckpt', '-zckpt', type = str, default = "/srv/storage/datasets/rafaelvieira/text2expression/should_be_good/Zs.pkl")
    parser.add_argument('--zsent_ckpt', '-zsckpt', type = str, default = "/srv/storage/datasets/rafaelvieira/text2expression/should_be_good/Zsent.pkl")

    args = parser.parse_args()
    main(args)
