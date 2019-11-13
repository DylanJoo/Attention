import torch
import torch.utils.data as Data  # Dataset format in pytorch
from torch.autograd import Variable  #?? meaning of variable.
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split as tts

import numpy as np
import util

class handler():

    def __init__(self, arg = [300, 64, 'cpu'], max_len = 256):
        self.n_batch = arg[1]
        self.max_len = max_len
    
    def split_tt(self, data, seg_size = (0.8, 0.0, 0.2), resample = False): #MUST shuffled

        x_all, lbl_all = data

        x_train, x_test, lbl_train, lbl_test = \
            tts(x_all, lbl_all, stratify = lbl_all, test_size = seg_size[2], random_state = 1)

        if resample == True:
            x_train, lbl_train = self.resample(x_train, lbl_train)
        
        return x_train, x_test, lbl_train, lbl_test


    def split_tvt(self, data, seg_size = (0.7, 0.1, 0.2), resample = False): # to a loader

        val_s = seg_size[1]/(seg_size[0]+seg_size[1])

        x_train, x_test, lbl_train, lbl_test = \
                 self.split_tt(data, seg_size)
                 
        x_train, x_val, lbl_train, lbl_val = \
                 tts(x_train, lbl_train, test_size = val_s, stratify = lbl_train, random_state = 1)

        if resample == True:
            x_train, lbl_train = self.resample(x_train, lbl_train)
        
        train_loader = self.load(x_train, lbl_train)
        val_loader = self.load(x_val, lbl_val)
        test_loader = self.load(x_test, lbl_test)

        return train_loader, val_loader, test_loader

    def load(self, x_, lbl_, dtype = [torch.LongTensor, torch.LongTensor]):

        x_ = util.np2list(x_)
        x_ = util.padding(x_, self.max_len) # fill the x with same length to train the NN model.

        #TENSORIZE.
        feature = Variable(dtype[0](x_))
        label = Variable(dtype[1](lbl_))
                
        dataset = Data.TensorDataset(feature, label)

        print("Data size: ", len(dataset))
        #PACK...into DataLoader obj.(which can implement batching feed)
        data_loader = Data.DataLoader(dataset, self.n_batch)

        return data_loader

    def resample(self, x, y, mode = 'ros'):
        #Resampling(over/undersampling)
        if mode == 'ros':
            x_re, y_re = util.oversampling(x.reshape(-1, 1), y)
        elif mode == 'rus':
            x_re, y_re = util.undersampling(x.reshape(-1, 1), y)
        print('RESAMPLED: '+ len(x)+ 'to', len(x_re))
        return x_re.reshape(-1, ), y_re


        
