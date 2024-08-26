import torch
import torch.nn as nn
import math

class MultiHeadedAttention(nn.Module):

    def __init__(self, num_heads: int = 1, size: int = 768, dropout: float = 0.1, k = 8):

        super(MultiHeadedAttention, self).__init__()

        assert size % num_heads == 0

        self.head_size = head_size = size // num_heads
        self.model_size = size
        self.num_heads = num_heads

        self.k_layer = nn.Linear(size, num_heads * head_size)
        self.v_layer = nn.Linear(size, num_heads * head_size)
        self.q_layer = nn.Linear(size, num_heads * head_size)

        self.output_layer = nn.Linear(size, size)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.k = k

    def forward(self, k: torch.Tensor, v: torch.Tensor, q: torch.Tensor, mask: torch.Tensor = None, padding_mask: torch.Tensor = None):

        batch_size = k.size(0)
        num_heads = self.num_heads

        # project the queries (q), keys (k), and values (v)
        k = self.k_layer(k)
        v = self.v_layer(v)
        q = self.q_layer(q)

        # reshape q, k, v for our computation to [batch_size, num_heads, ..]
        k = k.view(batch_size, -1, num_heads, self.head_size).transpose(1, 2)
        v = v.view(batch_size, -1, num_heads, self.head_size).transpose(1, 2)
        q = q.view(batch_size, -1, num_heads, self.head_size).transpose(1, 2)

        # compute scores
        q = q / math.sqrt(self.head_size)

        # batch x num_heads x query_len x key_len
        scores = torch.matmul(q, k.transpose(2, 3))

        # apply the mask (if we have one)
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(1), float('-inf'))

        # apply attention dropout and compute context vectors.
        attention = self.softmax(scores)
        attention = self.dropout(attention)
        attention = attention.squeeze(1)

        indices = torch.topk(attention, k = self.k, sorted = False).indices
        src = torch.ones(indices.shape, device = attention.device)
        result = torch.zeros(indices.size(0), 256, 256, dtype=src.dtype, device = attention.device)
        result.scatter_add_(1, indices, src)
        count = torch.sum(result, dim = 2)
        s_indices = torch.topk(count, k = self.k).indices
        return s_indices

class SentimentEncoder(torch.nn.Module):
    def __init__(
        self,
        bert,
        s_bert,
        device
    ):
        
        super().__init__()
        self.device = device        
        self.bert = bert.to(self.device)
        self.s_bert = s_bert.to(self.device)
        self.attention = MultiHeadedAttention().to(self.device)

        for p in s_bert.parameters():
            p.requires_grad = False

        for p in bert.parameters():
            p.requires_grad = False
        
    def forward(self, t_text, r_text):
        tokens_features = self.bert(t_text)[0]
        sentence_features = self.s_bert.encode(r_text, convert_to_tensor = True)
        topk_att = self.attention(tokens_features, tokens_features, tokens_features)
        topk_att = topk_att.sort().values
        selected_features = self.batched_index_select(tokens_features, 1, topk_att)
        sentence_features = sentence_features.unsqueeze(1).repeat(1, selected_features.size(1), 1)
        output = torch.cat([selected_features, sentence_features], dim = 2)
        return output

    def batched_index_select(self, t, dim, inds):
        dummy = inds.unsqueeze(2).expand(inds.size(0), inds.size(1), t.size(2))
        out = t.gather(dim, dummy) # b x e x f
        return out