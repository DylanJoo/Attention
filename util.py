import re
import sys
import time
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np

def timeit():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    
def hackssl():
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    
def sent2words(sent):
    return np.array(word_tokenize(sent))

def strnormalized(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    #e.g. $%^#@&...
    string = re.sub(r"\s\'", " ", string)
    #e.g. 'the, 'this ...specially designed for wrong quotation marks!
    string = re.sub(r"\'m", " \'m", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string) 
    string = re.sub(r"\'d", " \'d", string) 
    string = re.sub(r"\'ll", " \'ll", string)
    #e.g. am, are, have, will...
    string = re.sub(r" : ", ":", string)
    string = re.sub(r",", " , ", string) 
    string = re.sub(r"!", " ! ", string) 
    string = re.sub(r"\(", " ( ", string) 
    string = re.sub(r"\)", " ) ", string) 
    string = re.sub(r"\?", " ? ", string)
    #e.g. marks normalized
    string = re.sub(r"\s{2,}", " ", string)
    # convert 2+ space into only one
    return string.strip().lower()

def swremoval(lst):
    stop = set(stopwords.words('english'))
    output = np.array([word for word in lst if word not in stop])
    return output

def list2file(file, X, Y):
    with open(file.join('-x'), 'w') as f:
            for i in range(len(X)):
                    f.write('{}\n'.format(X[i]))
            f.close()

    with open(file.join('-lbl'), 'w') as f:
            for i in range(len(Y)):
                    f.write('{}\n'.format(Y[i]))
            f.close()

def padding(x, size = -1):
    from statistics import mode, mean
    
    if size < 0:
            size = mean(map(len, x))
            # if assign a value, all the X would be the same length
            # zero for frequntly appeared length as padding.
            
    return [i+int(size-len(i))*[0] if len(i) < size 
            else i[:size] for i in x]

def cutting(x, size = 10):
    if len(x) < size:
        for i in range(size-len(x)):
            x = np.append(x, x[i])
    else:
        x = x[:10]
    return x

def np2list(np_x):
    x = []
    for doc in np_x:
        doclist = [word for word in doc]
        x.append(doclist)
    return x

def undersampling(x, lbl): # apply imblearn pack.
        from imblearn.under_sampling import RandomUnderSampler
        from sklearn.utils import shuffle
        rus = RandomUnderSampler(random_state = 87)
        x_res, y_res = rus.fit_resample(x, lbl)
        return shuffle(x_res, y_res, random_state = 87)

def oversampling(x, lbl): # apply imblearn pack.
        from imblearn.over_sampling import RandomOverSampler
        from sklearn.utils import shuffle
        ros = RandomOverSampler(random_state = 87)
        x_res, y_res = ros.fit_resample(x, lbl)
        return shuffle(x_res, y_res, random_state = 87)
