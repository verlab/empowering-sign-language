from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import Transformer
import math

class PositionalEncoding(nn.Module):
    def __init__(
        self,
        emb_size: int,
        dropout: float,
        maxlen: int = 5000
    ):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

class SentiTransformerEncoder(nn.Module):
    def __init__(
        self, 
        d_model = 768, 
        nhead = 8, 
        num_layers = 8, 
        src_vocab_size = 19, 
        emb_size = 768, 
        dropout = 0.1
    ):
        super(SentiTransformerEncoder, self).__init__()
        self.encoder_layer = torch.nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = torch.nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(
            emb_size, 
            dropout=dropout
        )

    def forward(
        self,
        src: Tensor,
        src_mask: Tensor,
        src_key_padding_mask: Tensor
    ):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        outs = self.transformer_encoder(
            src_emb, 
            mask = src_mask, 
            src_key_padding_mask = src_key_padding_mask
        )
        return outs