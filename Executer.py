from Handler import handler
import torch
import numpy as np
from earlyStopping import EarlyStopping
import time

class executer():

    def __init__(self, model_, model_name, arg = [300, 64, 'cpu'], prt_model = False):
        self.model = model_

        if prt_model != False:
            self.model.load_embedding(prt_model)
            
        self.model_name = model_name

        self.n_epoch = arg[0]
        self.device = arg[2]
        self.hist = {}

    def train(self, train_loader, val_loader, case = None):

        es = EarlyStopping(patience=10, verbose=True)

        optimizer = torch.optim.Adam(self.model.parameters())
        #optimizer = torch.optim.SGD(self.model.parameters(), lr=3e-4, momentum=0.9)	
        criterion = self.model.loss_function()
        
        self.model.to(self.device)

        if case != None:
            self.path = 'case'+str(case)+'/'+self.model_name
        else:
            self.path = self.model_name

        for epoch in range(self.n_epoch):
            start = time.time()
    
            self.model.train()   #Optional? preventing overfitting??
            
            train_loss, valid_loss = 0.0, 0.0
            train_acc = 0.0
            for i, (x_, lbl_) in enumerate(train_loader):
                    x_ = x_.to(self.device)
                    lbl_ = lbl_.to(self.device)
                    
                    optimizer.zero_grad()
                    
                    pred = self.model(x_)  # FP
                    #print('pr:', pred)
                    # pred with a vectors
                    loss = criterion(pred, lbl_)  # Loss counting'
                    loss.backward() # BP
                    #print('pred:', np.array(list(map(np.argmax, pred.detach().cpu()))))
                    #print('lbl:', np.array(lbl_))
                    
                    train_loss += loss.item()*len(x_)/len(train_loader.dataset)
                    train_acc += self.accuracy(pred, lbl_).item()*len(x_)/len(train_loader.dataset)
                    optimizer.step() # UPDATE
            print('pred:', np.array(list(map(np.argmax, pred.detach().cpu()))))
            print('lbl:', np.array(lbl_))
            self.hist['Epoch'] = epoch+1
            self.hist['time'] = time.time()-start
            self.hist['train_loss'] = train_loss		
            self.hist['train_acc'] = train_acc

            torch.save(self.model.state_dict(), self.path)
            
            if val_loader != None:
                    self.validate(val_loader)
                    val_loss = self.hist['val_loss']

                    es(val_loss, self.model, self.model_name)
                    
            self.print_hist()
            
            if es.early_stop:

                print('Early stopping')
                break

    def validate(self, val_loader):

            criterion = self.model.loss_function()

            self.model.load_state_dict(torch.load(self.path))
            self.model.to(self.device)

            self.model.eval()
            
            val_loss, val_acc = 0.0, 0.0

            lbl_pred, lbl_true = [], []
            with torch.no_grad():
                    for i, (x_, lbl_) in enumerate(val_loader):
                            x_ = x_.to(self.device)
                            lbl_ = lbl_.to(self.device)
                            
                            logit = self.model(x_)
                            loss = criterion(logit, lbl_)

                            #print(np.argmax(logit, axis = 1))
                            #print(lbl_)
                            val_loss += loss.item()*len(x_)
                            val_acc += self.accuracy(logit, lbl_).item()*len(x_) # of correctly predict.

                            lbl_pred += np.argmax(logit.detach(), axis = 1).tolist()
                            lbl_true += lbl_.tolist()
                            #print(lbl_pred)
                            #print(lbl_true)

            self.hist['val_loss'] = val_loss/len(val_loader.dataset)
            self.hist['val_acc'] = val_acc/len(val_loader.dataset)
            self.f1 = self.f1score(lbl_true, lbl_pred)
            
            #return lbl_true , lbl_pred, val_loss/len(val_loader.dataset)
        
    def print_hist(self):
        string = '===> '

        for k in self.hist.keys():
                val = self.hist[k]

                if type(val) == float:
                        string += '{}: {:.6f}\t'.format(k, val)	
                else:
                        string += '{}: {}\t'.format(k, val)
        print(string)

    def accuracy(self, pred, truth):  #input tensors
        return (np.array(list(map(np.argmax, pred.detach().cpu()))) ==\
                np.array(truth.cpu())).sum()/len(pred)

    def f1score(self, pred, truth):
        from sklearn.metrics import f1_score # so lazy
        return f1_score(truth, pred, average='macro')  

