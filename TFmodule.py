from torch import nn
from TFutils import clones, gen_mask
import torch.nn.functional as F
from torch.nn import LayerNorm, MultiheadAttention, Dropout

class Encoder(nn.Module):

    def __init__(self, layer_en, N=1, norm=None): #norm layer
        super(Encoder, self).__init__()

        self.layers = clones(layer_en, N) #Stack N layer
        self.norm = norm


    def forward(self, src, mask_list, kmask_list=None):
        #src is the sequence(source)
        #Stacking N
        for layer in self.layers:
            output = layer(src, mask_list[0])
        if (self.norm):
            return self.norm(output)
        else:
            return output

class EncoderLayer(nn.Module):

    def __init__(self, embed_dim, self_attn, ff_1, ff_2, do_rate=0.1):
        super(EncoderLayer, self).__init__()

        self.self_attn = self_attn #MultiheadAttention(n_head, embed_dim)
        
        self.ff_1, self.ff_2 = ff_1, ff_2
        self.norm_1, self.norm_2 = LayerNorm(embed_dim), LayerNorm(embed_dim)
        self.drop_1, self.drop_2 = Dropout(do_rate), Dropout(do_rate)
        
        self.drop_0 = Dropout(do_rate)
        
    def forward(self, src, src_mask=None, src_kmask=None):
        #Just let the mask be None for now.
        #Sublayer 1st: #self-ATTN, Default no keymask
        src_c = self.self_attn(src, src, src, attn_mask=src_mask)[0]

        src = self.norm_1(src + self.drop_1(src_c))
        
        #Sublayer 2nd: #FFN
        src_c = self.ff_2(self.drop_0(F.relu(self.ff_1(src))))

        #Out!
        src = self.norm_2(src + self.drop_2(src_c))
        return src

################################################################################

class Decoder(nn.Module):

    def __init__(self, layer_de, N=1, norm=None):
        super(Decoder, self).__init__()

        self.layers = clones(layer_de, N) #stack N layer
        self.norm = norm
        
    def forward(self, tgt, memory, mask_list, kmask_list=None):
        #tgt is the output sequence(target
        for layer in self.layers:
            output = layer(tgt, memory, mask_list[1], mask_list[2])
        if (self.norm):
            return self.norm(output)
        else:
            return output
    
class DecoderLayer(nn.Module):

    def __init__(self, embed_dim, self_attn, mh_attn, ff_1, ff_2, do_rate=0.1):
        super(DecoderLayer, self).__init__()

        self.self_attn = self_attn
        self.mh_attn = mh_attn
        #Actually, Both multiheadAttn

        self.ff_1, self.ff_2 = ff_1, ff_2
        self.drop_1, self.drop_2, self.drop_3 = Dropout(do_rate), Dropout(do_rate), Dropout(do_rate)
        self.norm_1, self.norm_2, self.norm_3 = LayerNorm(embed_dim), LayerNorm(embed_dim), LayerNorm(embed_dim)

        self.drop_0 = Dropout(do_rate)

    def forward(self, tgt, memory,
                tgt_mask=None, memory_mask=None,
                tgt_kmask=None, memory_kmask=None): #Keypadding mask
 
        #Sublayer 1st: #Self-ATTN, Default no keymask
        tgt_c = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask)[0]
        tgt = self.norm_1(tgt + self.drop_1(tgt_c))

        #Sublayer 2nd: #Masked MH-ATT (input the selfattn's properties), Default no keymask
        tgt_c = self.mh_attn(tgt, memory, memory, attn_mask=memory_mask)[0]
        tgt = self.norm_2(tgt + self.drop_2(tgt_c))
        
        #Sublayer 3nd: #FFN
        tgt_c = self.ff_2(self.drop_0(F.relu(self.ff_1(tgt))))

        #Out!
        tgt = self.norm_3(tgt + self.drop_3(tgt_c))
        return tgt

'''
#Residual networks
class SubLayer(nn.Module):
    
    def __init__(self, embed_dim, do_rate=0.1):
        super(SubLayer, self).__init__()
        self.norm = LayerNorm(embed_dim)
        self.drop = nn.Dropout(do_rate)
        
    def forward(self, x, sub=None):
        #LayerNorm(x+sublayer(x)) ATTNETION IS ALL U NEED.
        return self.norm(x+self.drop(sub(x)))
'''
