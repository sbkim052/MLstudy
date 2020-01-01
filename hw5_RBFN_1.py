# -*- coding: utf-8 -*-
"""
Machine Learning HW3 : implementing Kmeans algorithm 
@author: 21500080 Sungbin Kim 
"""
import six.moves.cPickle as pickle
import gzip
import os
import numpy as np
import matplotlib.pyplot as plt
import operator

class RBFN():
    def __init__(self,rbf_neuron,sigma):
        self.rbf_neuron = rbf_neuron
        self.sigma = sigma
        self.centers = None
        self.weight = None
    
    def RBF_neuron(self,cen,data):
        #Calculate each neuron
        val=np.exp(-(1/(self.sigma))*np.linalg.norm(cen-data)**2)
        return val
    
    def calculate_phi(self,data):
        G=np.zeros((len(data),self.rbf_neuron))
        for data_index,data_point in enumerate(data):
            for cen_index,cen in enumerate(self.centers):
                G[data_index,cen_index]=self.RBF_neuron(
                        self.centers[cen_index],data_point)

        return G
    
    def train(self,data_x,label,cen):
        self.centers=cen
        psedo=self.calculate_phi(data_x)
        self.weight=np.dot(np.linalg.pinv(psedo),label) 
        
    def predict(self,X):
        G = self.calculate_phi(X)
        predictions = np.dot(G, self.weight)
        predictions = self.threshold(predictions)
        
        return predictions
    
    def threshold(self,predict):
        for index,data in enumerate(predict):
            if(data>0.5):
                predict[index]=1                
            else:
                predict[index]=0
        return predict
                
        
        

def load_data(data):
    list=np.array([])
    slice=[]
    f=open(data,'r')
    while True:
        line=f.readline()
        if not line: break
        slice_arr=np.array([])
        slice=(line.split())
        for i in range(len(slice)):
            slice_arr=np.append(slice_arr,float(slice[i]))
            
        list=np.append(list,slice_arr,axis=0)
        #list=list+slice_arr
    list=list.reshape(int(len(list)/len(slice_arr)),len(slice_arr))

    f.close()
    
    return list
    
def normalization(train,maxval,minval):
    data=(train-minval)/(maxval-minval)
    return data

def distance(A,B):
    vec_dist=np.sum((A-B)**2)**(1/2)
    return vec_dist

def kmeans(train,k):
    iter=300
    centroid={}
    cluster={}
    pre_err=np.inf
    cur_err=0
    #initialize centroid with the train set
    for i in range(k):
        centroid[i]=train[i]
        #centroid[i]=np.random.rand(len(train[0]))     
            
    #EM algorithm
    for t in range(iter):
        #initialize number of clusters
        for i in range(k):
            cluster[i]=[]            
        #cluster the data into min(dist)        
        for r in train:
            dist=[]
            for num in range(k):
                dist.append(distance(r,centroid[num]))
            centroid_index=dist.index(min(dist))
            cluster[centroid_index].append(r)
        
        #centroid update with the mean
        for i in range(k):
            centroid[i]=np.average(cluster[i],axis=0)
            
        #evaluate pre_err with cur_err
        cur_err=measure_error(cluster,centroid,k)
        
        if (pre_err-cur_err)<0.01:
            update=t
            break
        pre_err=cur_err
            
    return cluster, centroid, cur_err, update

def measure_error(cluster,centroid,k):
    error_sum=0
    for i in range(k):
        for r in cluster[i]:
            error_sum+=np.power(distance(r,centroid[i]),2)
    return error_sum
 
def scatter(x, label,prediction):
    #x=np.sort(x)
    #label=np.sort(label)
    one=np.where(label==1)
    zero=np.where(label==0)
    pone=np.where(prediction==1)
    
    
    plt.scatter(x[one][:,0],x[one][:,1],alpha=0.3,color='red')
    plt.scatter(x[pone][:,0],x[pone][:,1],alpha=0.3,color='blue')
    plt.scatter(x[zero][:,0],x[zero][:,1],color='white')

        #plt.lines=(xval,label[index],type='l')
    plt.show() 

def accuracy(prediction,label):
    acc=0
    for i in range(len(label)):
        if prediction[i]==label[i]:
            acc+=1
    
    acc=(acc/len(label))*100
    return acc

def cal_var(X):
    mean=X.mean()
    var=0
    for i in range(len(X)):
        var+=np.linalg.norm(X[i]-mean)**2
    var/=len(X)
    return var

if __name__ == '__main__':
    train_set1=load_data("cis_train1.txt")
    train_set2=load_data("cis_train2.txt")
    test_set=load_data("cis_test.txt")
    #find the center by kmeans algorithm
    #cluster1, centroid1, error1, update1=kmeans(train_set1[:,0:-1],9)
    cluster2, centroid2, error2, update2=kmeans(train_set1[:,0:-1],59)
    
    #var1=cal_var(train_set1[:,0:-1])
    var2=cal_var(train_set2[:,0:-1])
    
    #r1=RBFN(9,1)
    r2=RBFN(59,1)
    
    #r1.train(train_set1[:,0:-1],train_set1[:,-1],centroid1)
    r2.train(train_set2[:,0:-1],train_set2[:,-1],centroid2)

    
    #Y1=r1.predict(test_set[:,0:-1])
    Y2=r2.predict(test_set[:,0:-1])
    
    #scatter(test_set[:,0:-1],test_set[:,-1],Y1)
    #print("accuracy of trainset1 : ", accuracy(Y1,test_set[:,-1]))
    scatter(test_set[:,0:-1],test_set[:,-1],Y2)
    print("accuracy of trainset2", accuracy(Y2,test_set[:,-1]))

    
    
    