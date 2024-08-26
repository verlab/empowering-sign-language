import os
import sys
from venv import create
import torch
import torchvision

PAD_IDX = 0

def create_mask(src):
    src_seq_len = src.shape[0]
    src_mask = torch.zeros((src_seq_len, src_seq_len))
    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    return src_mask, src_padding_mask


def main():
    input = torch.LongTensor([[1,2,4,5,6,0,0],[4,3,2,9,0,0,0]]).transpose(1, 0)
    src_mask, padding_mask = create_mask(input)
    embedding = torch.nn.Embedding(10, 512)
    emb_input = embedding(input)
    encoder_layer = torch.nn.TransformerEncoderLayer(d_model=512, nhead=8)
    transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=6)
    out = transformer_encoder(emb_input, mask = src_mask, src_key_padding_mask = padding_mask)
    import pdb
    pdb.set_trace()

if __name__ == "__main__":
    main()