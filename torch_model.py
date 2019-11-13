import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm, MultiheadAttention, Dropout
import util
from TFmodule import Encoder, Decoder, EncoderLayer, DecoderLayer
from copy import deepcopy
from TFutils import *

class Transformer(nn.Module):
    
    def __init__(self, n_head=2, d_head = 2, embed_dim=100, N_en=6, N_de=6,
                  classes = 2, ff_dim=2048, do_rate=0.1, max_len=256,
                  activation="relu", custom_encoder=None, custom_decoder=None,
                  masks=[False, False, False], kmasks=[False, False, False]):
        
        super(Transformer, self).__init__()

        #===Base model(attn, enc, dec, ff)
        mhattn = MultiheadAttention(embed_dim, n_head)
        selfattn = MultiheadAttention(embed_dim, n_head)
        ff_1 = nn.Linear(embed_dim, ff_dim)
        ff_2 = nn.Linear(ff_dim, embed_dim)
        position = PositionalEncoding(embed_dim, do_rate)

        #===Masked attention(for seqs/keys) #src, tgt, memory
        self.masks = masks
        self.kmasks = kmasks

        #===Main Archetecture(enc, dec)
        self.encoder = Encoder(
            EncoderLayer(embed_dim, deepcopy(mhattn), deepcopy(ff_1), deepcopy(ff_2), do_rate), N_en)
        self.decoder = Decoder(
            DecoderLayer(embed_dim, deepcopy(selfattn), deepcopy(mhattn), deepcopy(ff_1), deepcopy(ff_2), do_rate), N_de)

        #===Embedding setting(src, tgt)
        self.src_embed = nn.Sequential(nn.Embedding(10000, embed_dim), deepcopy(position))
        self.tgt_embed = nn.Sequential(nn.Embedding(10000, embed_dim), deepcopy(position))

        #===Fianl FC
        self.final = nn.Linear(embed_dim*max_len, classes)

        #===Loss function definition
        self.loss = nn.CrossEntropyLoss()

        #===Parameters
        self.embed_dim = embed_dim
        self.max_len = max_len
        
    def load_embedding(self, model):
        weight = model.vectors
        self.src_embed = nn.Embedding.from_pretrained(torch.FloatTensor(weight))
        self.tgt_embed = nn.Embedding.from_pretrained(torch.FloatTensor(weight))
        
    def loss_function(self):
        return self.loss

    def forward(self, src_, tgt_=None):

        #Not sure if it is right in text classification on Attention model.
        if tgt_ == None:
            tgt_ = src_

        #length of src, tgt, memory
        s, t = src_.size(0), tgt_.size(0)
        length = [(s,s), (t,t), (t,s)]
        masks = []

        #Build masks(no key mask for now)
        for i in range(3):
            if self.masks[i]:
                masks.append(gen_mask(length[i]))
            else:
                masks.append(None)

        #Embedding the sentenses
        embed_src = self.src_embed(src_)
        embed_tgt = self.tgt_embed(tgt_)
        
        memory = self.encoder(embed_src, masks)
        output_ = self.decoder(embed_tgt, memory, masks)
        #Ignore the kmask for now.

        output = self.final(output_.view(-1, self.embed_dim*self.max_len)) #100 -> 2
        
        return output




#For Now, Use the default pytorch mhattn layer ;)#
class MultiHeadAtt(nn.Module):
    def __init__(self, n_head=2, embed_dim=64, d_head=2, classes=2, max_len=256):
        super(MultiHeadAtt, self).__init__()
        self.loss = nn.CrossEntropyLoss()
        self.embed = nn.Embedding(10000, embed_dim)
        
        self.n_head = n_head
        self.d_model = embed_dim
        self.d_head = d_head
        self.max_len = max_len

        self.q_net = nn.Linear(embed_dim, n_head*d_head, bias=False)
        # (seq ,embed) * [(embed, (q1~q2))] = (seq, (q1~q2))
        
        self.kv_net = nn.Linear(embed_dim, 2*n_head*d_head, bias=False)
        # (seq, (k1~k2) - (v1~v2))

        self.o_net = nn.Linear(n_head*d_head, embed_dim, bias=False)

        self.scale = 1/(d_head ** 0.5)
        self.drop = nn.Dropout(p=0.2)

        self.final = nn.Linear(embed_dim*max_len, classes)

    def load_embedding(self, model):
        weight = model.vectors
        self.embed = nn.Embedding.from_pretrained(torch.FloatTensor(weight))

    def loss_function(self):
        return self.loss

    def forward(self, x_):        
        x_fp = self.embed(x_)
        head_q = self.q_net(x_fp)
        head_k, head_v = torch.chunk(self.kv_net(x_fp), 2, -1) # reversed-cat
        
        head_q = head_q.view(x_fp.size(0), x_fp.size(1), self.n_head, self.d_head)
        head_k = head_k.view(x_fp.size(0), x_fp.size(1), self.n_head, self.d_head)
        head_v = head_v.view(x_fp.size(0), x_fp.size(1), self.n_head, self.d_head)
        # 3 dim to 4 dim

        att_score = torch.einsum('ibnd,jbnd->ijbn', (head_q, head_k))
        # count the attention score 
        att_score.mul_(self.scale)
        # Shrink the size
        att_prob = F.softmax(att_score, dim=1)
        # counting the attention weight.                
        att_vec = torch.einsum('ijbn, jbnd->ibnd', (att_prob, head_v))
        # get the final value
        
        att_vec = att_vec.contiguous().view(att_vec.size(0), att_vec.size(1), self.n_head*self.d_head)

        att_out = self.o_net(att_vec) + x_fp #??
        #word embedding + attention embedding?!

        output = self.final(att_out.view(-1, self.d_model*self.max_len))
        # flatten sum of(attention embedding & word embeddings)
        return output
'''
class LSTM(nn.Module):
    def __init__(self, embed_dim=64, hidden_dim=6, classes = 2):
        super(LSTM, self).__init__()
        self.loss = nn.CrossEntropyLoss()
        self.embed = nn.Embedding(10000, embed_dim)

        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first = True)
        self.final = nn.Linear(hidden_dim, classes)

        self.h_dim = hidden_dim
    def load_embedding(self, model):
        weight = model.vectors
        self.embed = nn.Embedding.from_pretrained(torch.FloatTensor(weight))

    def loss_function(self):
        return self.loss
	
    def forward(self, x_): # (batch, length)
        x_fp = self.embed(x_)   # (batch, length, embed_dim)

        lstm_out, _ = self.lstm(x_fp)
        # input size: (bathc, length, embed_dim)
        # O, h_s, c_s: (batch, length, hidden_dim)

        #print(lstm_out.size())
        out = self.final(lstm_out[:, -1, :])

        return out
'''
