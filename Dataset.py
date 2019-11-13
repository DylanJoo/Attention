import pandas as pd
import numpy as np
import util

class rt_data(): # rotten tomatoes

    def __init__(self, mode = None):
        self.x = []
        self.y = []
        if (mode == None):
            self.load()
        elif (mode == 'rand'):
            self.rand_load()
        else:
            self.cut_load()

    def load(self):
        p = './data/rt_polarity_all.txt'
        data = open(p, encoding='utf-8', errors='ignore').readlines()

        for doc in data:
            y_, x_ = doc.split(' ', 1)

            self.x.append(self._pp(x_))
            self.y.append(int(y_))

    def cut_load(self):
        p = './data/rt_polarity_all.txt'
        data = open(p, encoding='utf-8', errors='ignore').readlines()

        for doc in data:
            y_, x_ = doc.split(' ', 1)

            self.x.append(util.cutting(self._pp(x_)))
            self.y.append(int(y_))
        

    def rand_load(self):
        from random import shuffle
        p = './data/rt_polarity_all.txt'
        data = open(p, encoding='utf-8', errors='ignore').readlines()

        for doc in data:
            y_, x_ = doc.split(' ', 1)
            
            x_ = self._pp(x_)
            shuffle(x_)
            self.x.append(x_)
            self.y.append(int(y_))
    
    def _pp(self, x_):
        x = util.strnormalized(x_)
        x = util.sent2words(x)
        x = util.swremoval(x)
        return x


'''
Datasets:
-(19)Stanford sentiment(wo/ class 2,3,4),
-(36)AG News
-(179)IMDB large movie review
'''
