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
    def __init__(self,rbf_neuron):
        self.rbf_neuron = rbf_neuron
        self.sigma = None
        self.centers = None
        self.weight = None
    
    def RBF_neuron(self,cen,data,sig):
        val=np.exp(-(1/sig)*np.linalg.norm(cen-data)**2)
        return val
    
    def calculate_phi(self,data):
        G=np.zeros((len(data),self.rbf_neuron))
        for data_index,data_point in enumerate(data):
            for cen_index,cen in enumerate(self.centers):
                G[data_index,cen_index]=self.RBF_neuron(cen,data_point,self.sigma[cen_index])
        return G
        """
        for data_index in range(len(data)):
            for cen_index in range(len(self.centers)):
                G[data_index,cen_index]=self.RBF_neuron(self.centers[cen_index],data[data_index],self.sigma[cen_index])        
        return G
    
            G = np.zeros((len(X), self.hidden_shape))
        for data_point_arg, data_point in enumerate(X):
            for center_arg, center in enumerate(self.centers):
                G[data_point_arg, center_arg] = self._kernel_function(
                        center, data_point)
        return G
        """
    
    
    def fit(self,data_x,label,cen,var):
        self.centers=cen
        self.sigma=var
        psedo=self.calculate_phi(data_x)
        self.weight=np.dot(np.linalg.pinv(psedo),label)
        
    def predict(self,X):
        G = self.calculate_phi(X)
        predictions = np.dot(G, self.weight)
        return predictions
        

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

def variance(A,cen):
    print("\n\n\n\n")
    var={}
    for index,cenval in enumerate(cen):
        var[index]=0
        for data in A[index]:
            var[index]+=np.linalg.norm((data-cen[index]))**2
        var[index]=var[index]/len(A[index])
    return var

def cen_var(cen,var):
    for index, varval in enumerate(var):
        cen[index]=np.append(cen[index],var[index])
    return cen
    

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
        
        if (pre_err-cur_err)<0.001:
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
 
def scatter_circle(xy, label):
    for index,xyval in enumerate(xy):
        if label[index]>0.9:
            plt.scatter(xyval[0],xyval[1],alpha=0.5,color='blue')
        else:
            plt.scatter(xyval[0],xyval[1],alpha=0.5,color='green')
    plt.show() 

if __name__ == '__main__':
    #circle in square data
    #load the train data
    train_set=load_data("cis_train1.txt")
    test_set=load_data("cis_test.txt")
    #find the mu and variance
    cluster, centroid, error, update=kmeans(train_set[:,0:-1],5)
    var=variance(cluster,centroid)
    
    r=RBFN(5)
    r.fit(train_set[:,0:-1],train_set[:,-1],centroid,var)
    Y=r.predict(test_set[:,0:-1])
    
    scatter_circle(train_set[:,0:-1],train_set[:,-1])
    #scatter_circle(test_set[:,0:-1],Y)
    
    
    

    
    
    #train algorithm
    
    for i in range(1):
        RBF_train(w,train_set,centroid,var)
    
    for i in range(len(train_set)):
        plt.scatter(train_set[i][0],train_set[i][1],color='blue')
    plt.show()
    