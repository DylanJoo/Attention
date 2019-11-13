import gensim
from gensim.models import Word2Vec, KeyedVectors
import gensim.downloader as api
from gensim.test.utils import common_texts, get_tmpfile
import util
import numpy as np
from sklearn.utils import shuffle

class w2v_pretrained():

    def __init__(self, model_name):        
        try:
            util.hackssl()
            self.model = api.load(model_name) # import the gensim pretrained model
            # 'word2vec-google-news-300', 'glove-wiki-gigword-100d'
        except:
            self.model = KeyedVectors.load('./data/'+model_name+'.kv')
        self.add()
        
    def getDict(self):
        return self.model.vocab

    def add(self, token=None, vector=None):
        self.model.add('<ZERO>', np.zeros([1, self.model.vector_size]))
        self.model.add('<UNK>', np.random.randn(1, self.model.vector_size))
        #default input the <zero> & <unk>
        if token != None:
                self.model.add(token, vector)


class w2v_selftrained(): # need to be pre-split,

    def __init__(self, corpus, lbl, oc = False):
        x, _, lbl, _ = handler().split_tt((corpus, lbl))
        if oc == True:
            fil = (lbl == 1)
            x = x[fil]
            print('One class strategy.')
        x = util.np2list(x)
        self.model = Word2Vec(x, size = 100, window = 5, min_count = 10, iter = 3)
        self.add()

    def getDict():
        return self.model.vocab

    def add(self, token=None, vector=None):
        self.model.add('<ZERO>', np.zeros([1, self.model.vector_size]))
        self.model.add('<UNK>', np.random.randn(1, self.model.vector_size))
        #default input the <zero> & <unk>
        if token != None:
                self.model.add(token, vector)
