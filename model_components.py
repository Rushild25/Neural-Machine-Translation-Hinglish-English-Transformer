import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def text_preprocessing(text):
    import re
    from bs4 import BeautifulSoup
    text = text.lower()
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+'", "'", text)
    text = re.sub(r"'\s+", "'", text)
    return text.strip()

def word_tokenize(seq):
    return seq.split()

def adjust_seq(tokenized_sent, max_len=300):
    return ['<sos>'] + tokenized_sent[:max_len] + ['<eos>']

class Masking(nn.Module):
    def __init__(self, max_len):
        super().__init__()
        casaul_mask = torch.triu(
            torch.ones((max_len, max_len), dtype=torch.bool),
            diagonal=1
        )
        self.register_buffer("casaul_mask", casaul_mask)

    def padding_mask(self, seq, pad_idx=0):
        mask = (seq != pad_idx).unsqueeze(1).unsqueeze(2)
        return mask.bool()

    def decoder_mask(self, decoder_seq, pad_idx=0):
        seq_len = decoder_seq.size(1)
        pad_mask = (decoder_seq != pad_idx).unsqueeze(1).unsqueeze(2)
        data_leakage_mask = self.casaul_mask[:seq_len, :seq_len]
        combined_mask = pad_mask & ~data_leakage_mask.unsqueeze(0).unsqueeze(0)
        return combined_mask

class Multi_Head_Attention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.q_matrix = nn.Linear(d_model, d_model)
        self.k_matrix = nn.Linear(d_model, d_model)
        self.v_vectors = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        q = self.q_matrix(q).view(batch_size,-1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_matrix(k).view(batch_size,-1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_vectors(v).view(batch_size,-1, self.num_heads, self.d_k).transpose(1, 2)
        x = attention_mechanism(q, k, v, mask)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.out_linear(x)

def attention_mechanism(q, k, v, mask=None):
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == False, float('-inf'))
    attn = F.softmax(scores, dim=-1)
    return torch.matmul(attn, v)

class Feed_Forward_Neural_Network(nn.Module):
    def __init__(self, d_model, nodes):
        super().__init__()
        self.l1 = nn.Linear(d_model, nodes)
        self.act1 = nn.ReLU()
        self.l2 = nn.Linear(nodes, d_model)

    def forward(self, x):
        return self.l2(self.act1(self.l1(x)))

class EncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, nodes, dropout_rate):
        super().__init__()
        self.self_attn_layer = Multi_Head_Attention(d_model, num_heads)
        self.ff_nn_layer = Feed_Forward_Neural_Network(d_model, nodes)
        self.dropout = nn.Dropout(dropout_rate)
        self.norm_layer1 = nn.LayerNorm(d_model)
        self.norm_layer2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        attn_out = self.self_attn_layer(x, x, x, mask)
        x = self.norm_layer1(x + self.dropout(attn_out))
        ff_out = self.ff_nn_layer(x)
        return self.norm_layer2(x + self.dropout(ff_out))

class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, nodes, dropout_rate):
        super().__init__()
        self.masked_attn_layer = Multi_Head_Attention(d_model, num_heads)
        self.cross_attn_layer = Multi_Head_Attention(d_model, num_heads)
        self.ff_nn_layer = Feed_Forward_Neural_Network(d_model, nodes)
        self.dropout = nn.Dropout(dropout_rate)
        self.norm_layer1 = nn.LayerNorm(d_model)
        self.norm_layer2 = nn.LayerNorm(d_model)
        self.norm_layer3 = nn.LayerNorm(d_model)

    def forward(self, x, enc_out, self_mask=None, cross_mask=None):
        attn_out = self.masked_attn_layer(x, x, x, self_mask)
        x = self.norm_layer1(x + self.dropout(attn_out))
        cross_out = self.cross_attn_layer(x, enc_out, enc_out, cross_mask)
        x = self.norm_layer2(x + self.dropout(cross_out))
        ff_out = self.ff_nn_layer(x)
        return self.norm_layer3(x + self.dropout(ff_out))

class Encoder(nn.Module):
    def __init__(self, d_model, num_heads, nodes, dropout_rate, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderBlock(d_model, num_heads, nodes, dropout_rate) for _ in range(num_layers)
        ])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x

class Decoder(nn.Module):
    def __init__(self, d_model, num_heads, nodes, dropout_rate, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderBlock(d_model, num_heads, nodes, dropout_rate) for _ in range(num_layers)
        ])

    def forward(self, x, enc_out, self_mask=None, cross_mask=None):
        for layer in self.layers:
            x = layer(x, enc_out, self_mask, cross_mask)
        return x

class Transformer(nn.Module):
    def __init__(self, d_model, num_heads, nodes, dropout_rate, num_layers, vocab_size):
        super().__init__()
        self.encoder = Encoder(d_model, num_heads, nodes, dropout_rate, num_layers)
        self.decoder = Decoder(d_model, num_heads, nodes, dropout_rate, num_layers)
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, seq_A, seq_B, mask=None, self_mask=None, cross_mask=None):
        enc_out = self.encoder(seq_A, mask)
        dec_out = self.decoder(seq_B, enc_out, self_mask, cross_mask)
        return self.linear(dec_out)

class InputReady(nn.Module):
    def __init__(self, d_model, vocab_size, max_len=300):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        k = torch.exp(-math.log(10000.0) * torch.arange(0, d_model, 2)/d_model)
        pe[:, 0::2] = torch.sin(pos * k)
        pe[:, 1::2] = torch.cos(pos * k)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.embedding(x) + self.pe[:, :x.size(1), :]
