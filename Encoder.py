import numpy as np
import util
from itertools import chain
from sklearn.model_selection import train_test_split as tts
from gensim.models import KeyedVectors, Word2Vec


'''
Param:
    fit(corpus):2d-array
        fit all the vocabulary into a big dictionary.
        It would be used as the look-table to perform the term convertion.

    transform(encoded sequence):2d-array
Attr:

'''

class OneHotEncoder():

    def __init__(self):

        self.dict = {}  # for the purpose, pretrained dictionary entered
        self.n_vocab = len(self.dict)
    
    def fit_transform(self, corpus):

        vocab = list(set(chain.from_iterable(corpus)))
	#build a vocab dictionary from sets.

        self.n_vocab = len(vocab)
        #self.dict = {term: idx for idx, term in enumerate(vocab)}
        # to get the inner informatio of this corpus, build a look-up table.
        
        return np.array([list(map(vocab.index, termlist)) for termlist in corpus])

    def fit_transform_pretrained(self, corpus, dictionary):

        vocab = list(dictionary)
        vocab_ = list(set(chain.from_iterable(corpus)))
        # e.g. google-news-300, glove-wiki...
        oov = [word for word in vocab_ if word not in dictionary]
        print(len(oov))
        print('OOV ratio in total:', len(oov)/len(vocab_))

        return np.array([list(map(vocab.index, [i if i not in oov else '<UNK>' for i in termlist])) for termlist in corpus])
        #return np.array([list(map(vocab.index, [i for i in termlist if i not in oov])) for termlist in corpus])
