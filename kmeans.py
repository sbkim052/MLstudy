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

def load_data(dataset):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    
    copied from http://deeplearning.net/ and revised by hchoi
    '''

    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            os.path.split(__file__)[0],
            dataset
        )
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        from six.moves import urllib
        origin = (
            'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        )
        print('Downloading data from %s' % origin)
        urllib.request.urlretrieve(origin, dataset)

    print('... loading data')

    # Load the dataset
    with gzip.open(dataset, 'rb') as f:
        try:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        except:
            train_set, valid_set, test_set = pickle.load(f)
    # train_set, valid_set, test_set format: tuple(input, target)
    # input is a numpy.ndarray of 2 dimensions (a matrix)
    # where each row corresponds to an example. target is a
    # numpy.ndarray of 1 dimension (vector) that has the same length as
    # the number of rows in the input. It should give the target
    # to the example with the same index in the input.

    return train_set, valid_set, test_set

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
        #centroid[i]=train[i]
        centroid[i]=np.random.rand(len(train[0]))
        
            
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

def scatter_plot(cluster, centroid):
    palette=["red","blue","green","yellow","pink","cyan","magenta","purple","brown","gray"]
    for i in range(len(cluster)):
        for j in range(500):
            plt.scatter(cluster[i][j][0],cluster[i][j][1],alpha=0.5,color=palette[i])
        for j in range(len(centroid)):
            plt.scatter(centroid[j][0],centroid[j][1],alpha=1,color='black')
    plt.show()  
    
if __name__ == '__main__':
    #load the data
    train_set, val_set, test_set = load_data('mnist.pkl.gz')
    train_x, train_y = train_set 
    val_x, val_y = val_set
    test_x, test_y = test_set
    
    #find the 3, 9 labeld data
    index_num=np.where((3==train_y)|(9==train_y))
    train_x=train_x[index_num]
    
    #normalize the data
    train_norm_x=normalization(train_x,train_x.max(),train_x.min())
    """
    #implement kmeans in raw data
    for k in [2,3,5,10]:
        cluster,centroid,error,update=kmeans(train_norm_x,k)
        print("[rawdata]"," k=",k,"  error=",error, "(iteration: ",update,")")
    """
    #finding eigenvector and eigenvalue
    val,vec=np.linalg.eig(np.cov(train_x.T))   
    """
    for dim in [2]:
        for k in [10]:
            #project the data in to reduced dimension
            nvec=vec.T[0:dim]
            train_projected_x=np.matmul(train_norm_x,nvec.T)
            #normalization
            train_projected_x=normalization(train_projected_x,train_projected_x.max(),train_projected_x.min())
            #implement kmeans
            cluster, centroid, error, update=kmeans(train_projected_x,k)
            
            #print dim, k, error
            print("dim=",dim," k=",k,"  error=",error, "(iteration: ",update,")")
            #plot in 2D
            scatter_plot(cluster,centroid)
    """