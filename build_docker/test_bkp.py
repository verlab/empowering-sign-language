# -*- coding: utf-8 -*-
#for some reason this import has to come first, otherwise gives segfault
from transformers import BertTokenizer, BertModel
import argparse
import torch
import random
import numpy as np
import pickle
import os
import torch.nn as nn
from torchtext import data
from models.decoder_stgcn import Decoder
from data.dataset import SignProdDataset
import spacy
from torch import nn
from scipy import spatial

PAD_IDX = 0
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
max_input_length = tokenizer.max_model_input_sizes['bert-base-uncased']

def tokenize_and_cut(sentence):
    tokens = tokenizer.tokenize(sentence) 
    tokens = tokens[:max_input_length-2]
    return tokens

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

def read_sentences(dataset_root):
    sentences = list()
    files = os.listdir(dataset_root)
    files = sorted(files)
    for idx, file_ in enumerate(files[:50]):
        file_fp = os.path.join(dataset_root, file_)
        instance = read_pickle(file_fp)
        sentences.append((idx, instance))
    return sentences

def get_k_similar_sentences(embeddings, sentences, k = 2, key = "sent_embeddings"):
    sentences_dist = dict()
    for idx, sentence in sentences:
        embeddings_src = sentence[key]
        distance = spatial.distance.cosine(embeddings_src, embeddings)
    
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

def is_not_mean(mean_face, kps):
    for kp in kps:
        kp = np.asarray(kp)

        dist = np.linalg.norm(kp - mean_face)
        if dist > 0.10:
            return True
    return False


class NavigatorOnlySent(nn.Module):

    def __init__(
        self, 
        in_channels, 
        out_channels, 
        depth=1,
        device = None
      ):
        super(NavigatorOnlySent, self).__init__()
         
        self.model = nn.Sequential(
          nn.Linear(768, 3072),
          nn.Tanh(),
          nn.Linear(3072, 3072),
          nn.Tanh(),
          nn.Linear(3072, 2304),
          nn.Tanh(),
          nn.Linear(2304, 1536)
        )
       
    def forward(self, sent_feature, sem_feature):
        #x = torch.cat([sent_feature, sem_feature], dim = 1)
        x = self.model(sent_feature)
        return x

class NavigatorOnlySem(nn.Module):

    def __init__(
        self, 
        in_channels, 
        out_channels, 
        depth=1,
        device = None
      ):
        super(NavigatorOnlySem, self).__init__()
         
        self.model = nn.Sequential(
          nn.Linear(768, 3072),
          nn.Tanh(),
          nn.Linear(3072, 3072),
          nn.Tanh(),
          nn.Linear(3072, 2304),
          nn.Tanh(),
          nn.Linear(2304, 1536)
        )
       
    def forward(self, sent_feature, sem_feature):
        #x = torch.cat([sent_feature, sem_feature], dim = 1)
        x = self.model(sem_feature)
        return x


class Navigator(nn.Module):

    def __init__(
        self, 
        in_channels, 
        out_channels, 
        depth=1,
        device = None
      ):
        super(Navigator, self).__init__()
         
        self.model = nn.Sequential(
          nn.Linear(1536, 3072),
          nn.Tanh(),
          nn.Linear(3072, 3072),
          nn.Tanh(),
          nn.Linear(3072, 2304),
          nn.Tanh(),
          nn.Linear(2304, 1536)
        )
       
    def forward(self, sent_feature, sem_feature):
        x = torch.cat([sent_feature, sem_feature], dim = 1)
        x = self.model(x)
        return x

def project_l2_ball_torch(zt, device):
    unit_tensor = torch.Tensor([1]).to(device)
    return zt/torch.maximum(torch.sqrt(torch.sum(zt**2, axis = 1))[:, None], unit_tensor)

def main(args):

    SEED = 1234
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    device = torch.device("cpu")
    #mean_face = np.load("mean_face.npy")/256
    
    #dataset_root_train = args.dataset_root_train
    dataset_root_test = args.dataset_root_test
    output_root = args.outputs_root
    
    os.makedirs(output_root, exist_ok = True)

    net_z = Navigator(768*2,768*2).to(device)
    net_z.load_state_dict(torch.load("navigator_normal_new.pth", map_location=torch.device("cpu")))
    net_z.eval()

    decoder = Decoder(device).to(device)
    decoder.load_state_dict(torch.load(args.decoder_ckpt, map_location=torch.device("cpu")))
    decoder.eval()

    #print("Reading Zs")
    Z = read_pickle(args.zs_ckpt)

    files = sorted(os.listdir(dataset_root_test))
    #print("Reading sentences")
    #sentences = read_sentences(dataset_root_train)
    print("Running inference")
    for idx, file_ in enumerate(files):

        file_fp = os.path.join(dataset_root_test, file_)
        output_fp = os.path.join(output_root, file_.replace(".pkl", ".npz"))

        try:
            instance = read_pickle(file_fp)
        except:
            continue
    
        sem_embeddings = instance["sem_embeddings"]
        sent_embeddings = instance["sent_embeddings"]

        with torch.no_grad():

            sent_embeddings = torch.Tensor(sent_embeddings).unsqueeze(0).to(device)
            sem_embeddings = torch.Tensor(sem_embeddings).unsqueeze(0).to(device)

            input_net = net_z(sent_embeddings, sem_embeddings).to(device)

            input_net = input_net.reshape(-1, 768, 2)
            input_net_projected = project_l2_ball_torch(input_net, device)

            faces_fake_navigator_projected = decoder(None, None, input_net_projected).permute(0, 2, 3, 1).cpu().numpy()

            #print("plotting")
            #Utils.visualize_data_single(faces_fake_navigator_projected, "navigator_proj_{}.mp4".format(file_))
            save_predictions(output_fp, faces_fake_navigator_projected)
    print("Finished inference")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument('--device', '-dev', type = int, default = 0)
    parser.add_argument('--batch_size', '-bs', type = int, default = 42)
    parser.add_argument('--epochs', '-e', type = int, default = 200000)
    parser.add_argument('--dataset_root_test', '-dtr', type = str, default = "test_repro/test")
    parser.add_argument('--dataset_root_train', '-dte', type = str, default = "test_repro/test")
    parser.add_argument('--outputs_root', '-ot', type = str, default = "outputs_reprotest")
    parser.add_argument('--decoder_ckpt', '-dckpt', type = str, default = "embsmall_bet/decoder.pth")
    parser.add_argument('--zs_ckpt', '-zckpt', type = str, default = "embsmall_bet/Zs.pkl")

    args = parser.parse_args()
    main(args)