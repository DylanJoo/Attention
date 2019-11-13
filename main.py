from Executer import executer
from Encoder import OneHotEncoder as ohe
from Handler import handler
from Dataset import rt_data
from w2v import w2v_pretrained
from util import timeit
import torch_model as models
import numpy as np

arg = [300, 64, 'cpu']

##Pretrained model
#------
w2v = w2v_pretrained('glove-wiki-gigaword-100')
model = w2v.model
d = model.vector_size


#Define main function
#------
def run_nn(model, loaders, model_name, prt_model = False, case = None):
    print('==='+ model_name +'===<start>')
    
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('# of para: {}'.format(total))
    tr, val, te = loaders

    e = executer(model, model_name, arg, prt_model)
    e.train(tr, val, case)
    
    e.validate(te)
    print('Accuracy: {}'.format(e.hist['val_acc']))

    with open("case"+str(case)+'/'+model_name+".txt", "a") as f:
            print('Case:', str(case), file = f)
            print('Time:', str(timeit()), file = f)
            print('Accuracy: {}'.format(e.hist['val_acc']), file = f)
            print('F1-Score: {}\n'.format(e.f1), file = f)

##Experiments
#------
'''
Case 0: Original setting on datasets(rt)
'''
#x = rt.x
#lbl = rt.y

#x_encoded = ohe().fit_transform_pretrained(x, w2v.getDict())
#loaders = handler().split_tvt((x_encoded, lbl))

#run_nn(models.Transformer(2, 2, d, 1, 1, 2, 256), loaders, 'Transformer.pt', model, 0)
#run_nn(models.MultiHeadAtt(2, d, 8, 2), loaders, 'MultiHeadAtt.pt', model, 0)

'''
Case 1: Random-ordered datasets.

rt = rt_data('rand')
x = rt.x
lbl = rt.y

x_encoded = ohe().fit_transform_pretrained(x, w2v.getDict())
loaders = handler().split_tvt((x_encoded, lbl))

run_nn(models.MultiHeadAtt(2, d, 8, 2), loaders, 'MultiHeadAtt.pt', model, 1)
#run_nn(models.Transformer(2, 2, d, 1, 1, 2, 256), loaders, 'Transformer.pt', model, 1)
'''
'''
Case 2: Sentense-cutting
'''

rt = rt_data('cut')
x = rt.x
lbl = rt.y

mask_ = [True, True, True] # mask the decoder's self-attn
x_encoded = ohe().fit_transform_pretrained(x, w2v.getDict())
loaders = handler(arg, 10).split_tvt((x_encoded, lbl))

#run_nn(models.MultiHeadAtt(2, d, 8, 2, 10), loaders, 'MultiHeadAtt.pt', model, 2)
run_nn(models.Transformer(2, 2, d, 1, 1, 2, 256, 0.1, 10, masks=mask_), loaders, 'Transformer.pt', model, 2)

'''
Case 3: Multitasking(w/ Keyphrase)
'''

