#!/usr/bin/python
import numpy
import random
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
random.seed(777)

def ce(prob, labels):
   v=prob[:,labels]
   v=-numpy.log(v)
   return numpy.mean(v)

def accuracy(prob,labels):
   sel=numpy.argmax(prob,axis=1)
   return numpy.count_nonzero(sel == labels)/len(labels)

def distrib(labels,nlabels):
   lb=numpy.zeros(nlabels)
   for i in range(nlabels):
      lb[i]=numpy.count_nonzero(labels == i)
   if numpy.sum(lb)>0:
      lb=lb/numpy.sum(lb)
#   print(lb)
   return lb
   
class fitter():
   def __init__(self,maxn=10000):
      self.maxn=maxn
      return
   
   def fit(self,x,y):
      self.nlabels=max(y)+1
      samples,ndim=x.shape
      self.planes=[]
      limits=[[numpy.min(x[:,d]),numpy.max(x[:,d])] for d in range(ndim)]
      preds=numpy.zeros([samples, self.nlabels])
      old=-1e10
      for cyc in range(self.maxn):
         ind=random.choice(range(ndim))
         thr=random.uniform(limits[ind][0],limits[ind][1])
         mask=x[:,ind]>thr
         d1=distrib(y[mask],self.nlabels)
         d2=distrib(y[numpy.logical_not(mask)],self.nlabels)
         preds[mask,:]+=d1
         preds[numpy.logical_not(mask),:]+=d2
         ps=preds/numpy.sum(preds, axis=1)[:,None]
         score=accuracy(ps,y)#ce(ps,y)
         if score>old:
            old=score
            yhat=numpy.argmax(preds,axis=1)
 #           print(yhat)
            accu=sklearn.metrics.accuracy_score(y,yhat)
            print(score,accu,ce(ps,y))
         else:
            preds[mask,:]-=d1
            preds[numpy.logical_not(mask),:]-=d2
      return
   def predict(self,x):
      return



x=[[random.gauss(0,1),random.gauss(0,1)] for i in range(100)]
y=[0 for i in range(100)]
x.extend([[random.gauss(0,1)+3,random.gauss(0,1)] for i in range(100)])
y.extend([1 for i in range(100)])
x.extend([[random.gauss(0,1),random.gauss(0,1)+3] for i in range(100)])
y.extend([2 for i in range(100)])
#x.extend([[random.gauss(0,1)+3,random.gauss(0,1)+3] for i in range(100)])
#y.extend([0 for i in range(100)])

x=numpy.array(x)
y=numpy.array(y)

x_train,x_test, y_train,y_test = train_test_split(x,y,random_state=777)

#reference method
from sklearn.linear_model import LogisticRegression
model1=LogisticRegression(multi_class='multinomial',max_iter=500) # solver="saga")
model1.fit(x_train, y_train)
yp=model1.predict(x_test)
print ("LR train:",sklearn.metrics.accuracy_score(y_test,yp))
yp=model1.predict(x_train)
print ("LR test:",sklearn.metrics.accuracy_score(y_train,yp))

#plt.scatter(x_train[:,0],x_train[:,1],c=y_train)
#plt.show()

f=fitter()
f.fit(x_train,y_train)


