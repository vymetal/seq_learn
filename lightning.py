#!/usr/bin/env python
# coding: utf-8


from seqdata import getset
import sys
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
import matplotlib.pylab as plt
import numpy
from numpy import argmax, vstack
import random
import torch
from torch import Tensor
import pytorch_lightning as pl
from torch import optim, nn, utils
#from torch.utils.data import Dataset
from torch.utils.data import DataLoader
#from torch.utils.data import random_split
#from torch.nn import Linear,ReLU, LeakyReLU,ELU,Softmax,Module,CrossEntropyLoss
#from torch.optim import SGD,Adam
from torch.nn.init import kaiming_uniform_,xavier_uniform_, uniform_, ones_, zeros_, eye_
#from torch.nn.parameter import Parameter
import torchmetrics


def xsample(x0,size):
    res=[]
    while size-len(res)>=len(x0):
        res.extend(x0)
    res.extend(random.sample(x0,size-len(res)))
    return res



def prepare_data(cat_size=1000, batch_size=10):
    X,y=getset(7)
    X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=1234567, shuffle=True, stratify=y)

    u,c=numpy.unique(y_train, return_counts=True)
    print(f"Counts for minimal group: {numpy.min(c)}")
    print(f"Counts for maximal group: {numpy.max(c)}")
    cats=u

    test=[[numpy.array(x,dtype=numpy.float32),y] for x,y in zip(X_test,numpy.array(y_test,dtype=numpy.int64))]

    if cat_size>0:
        train=[]    
        #resample X_train and y_train
        #random.seed(1234567)
        new_X,new_y=[],[]
        for cat in cats:
            cat_xy=[(x,lab) for x,lab in zip(X_train,y_train) if lab==cat]
            qs=xsample(cat_xy, cat_size)
            qs=[[numpy.array(x,dtype=numpy.float32),numpy.array(lab,dtype=numpy.int64)] for x,lab in qs]
            train.extend(qs)
    else:
        train=[[numpy.array(x, dtype=numpy.float32),y] for x,y in zip(X_train,numpy.array(y_train,dtype=numpy.int64))]
   
    print(f'train data: {len(train)}, test data: {len(test)}, categories: {len(cats)}' )
    train_dl = DataLoader(train, batch_size=batch_size, shuffle=True)
    test_dl = DataLoader(test, batch_size=batch_size, shuffle=False)
    return train_dl, test_dl

class MLP(nn.Module):
    # define model elements
    def __init__(self, n_inputs):
        super(MLP, self).__init__()
        N=n_inputs*4
        self.bn=nn.BatchNorm1d(N)
        
        self.hidden1 = nn.Linear(n_inputs, N)
        self.act1 = nn.LeakyReLU()
        self.hidden2 = nn.Linear(N, N)
        self.act2 = nn.LeakyReLU()

        self.hidden_last = nn.Linear(N, 20)
        self.act_last = nn.Softmax(dim=-1)

        self.drop = nn.Dropout(0.5)

    # forward propagate input
    def forward(self, X):
        # input to first hidden layer
#        X = self.drop(X)
        X = self.hidden1(X)
        X = self.act1(X)
 #       X = self.drop(X)
        X=self.bn(X)
        X = self.hidden2(X)
        X = self.act2(X)
        #last
 #       X = self.drop(X)        
        X = self.hidden_last(X)
        X = self.act_last(X)
        return X


class PLModel(pl.LightningModule):
    def __init__(self,model):
        super().__init__()
        self.model=model
        self.lr=1.0e-3

    def training_step(self, batch, batch_idx):
        x, y = batch
        yhat = self.model(x)
        loss = nn.functional.cross_entropy(yhat, y)

        ind = numpy.argmax(yhat.detach().numpy(),axis=1)
        accuracy=accuracy_score(ind,y)
        #self.log("accuracy", accuracy, on_epoch=False, prog_bar=True)
        # Logging to TensorBoard by default
        #self.log("train_loss", loss)
        return {'loss':loss,'accuracy':torch.Tensor([accuracy,])}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        yhat = self.model(x)
        loss = nn.functional.cross_entropy(yhat, y)
        ind = numpy.argmax(yhat.detach().numpy(),axis=1)
        accuracy=accuracy_score(ind,y)
        #self.log("validation_accuracy", accuracy)
        # Logging to TensorBoard by default
        #self.log("validation_loss", loss)
        return {'validation_loss':loss, "validation_accuracy":torch.Tensor([accuracy,])}


    def training_epoch_end(self, training_step_outputs):
        #print(training_step_outputs)
        self.log("mtloss", torch.mean(torch.stack([x["loss"] for x in training_step_outputs])), prog_bar=True)
        self.log("mtacc", torch.mean(torch.stack([x["accuracy"] for x in training_step_outputs])), prog_bar=True)

    def validation_epoch_end(self, validation_step_outputs):
        self.log("vloss", torch.mean(torch.stack([x["validation_loss"] for x in validation_step_outputs])), prog_bar=True)
        self.log("vacc", torch.mean(torch.stack([x["validation_accuracy"] for x in validation_step_outputs])), prog_bar=True)
        pass
         
    def predict_step(self, batch, batch_idx, dataloader_idx = 0):
        return self(batch)

    def forward(self,x):
        #print(x)
        #print(x.shape)
        return self.model.forward(x)
    
    def configure_optimizers(self):
        #optimizer = optim.Adam(self.parameters(), lr=1e-3)
        #optimizer = optim.SGD(self.parameters(), lr=1e0, momentum=0.9)
        optimizer = optim.AdamW(self.parameters(), lr=1e-3)
        return optimizer




model=PLModel(MLP(161))
#model=torch.load("model_last")

train,test=prepare_data(1000,100)
print(len(train))
#model.dropout=nn.Dropout(0.5)
for cyc in range(1):
    trainer = pl.Trainer(limit_train_batches=1000, max_epochs=100,log_every_n_steps=1,auto_lr_find=False)
    trainer.fit(model=model, train_dataloaders=train, val_dataloaders=test)



tx=[x for x,_ in test][0]
ty=[cat for _,cat in test][0].detach().numpy()
print(ty)

predictions = trainer.predict(model, tx)
ind = [numpy.argmax(p) for p in predictions]
accuracy=accuracy_score(ind,ty)
print("Validation accuracy:", accuracy)

torch.save(model,"model_last")

