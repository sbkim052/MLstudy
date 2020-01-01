# -*- coding: utf-8 -*-
"""
Machine Learning HW3 : implementing KNN algorithm 
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



def knn_algorithm(train_data,test,label,k):
    distance=[]

    #calculate the distance    
    for i in range(len(train_data)):
        dist=np.sum((train_data[i,:]-test)**2)**1/2
        distance.append([dist,i])
    distance.sort()
        
    #maximum vote to select the best prediction
    max_class={}
    for i in range(k):
        select=label[distance[i][1]]
        max_class[select]=max_class.get(select,0)+1
    sorted_max_class=sorted(max_class.items(),
                            key = operator.itemgetter(1), reverse=True)
    prediction=sorted_max_class[0][0]
    
    return prediction    

if __name__ == '__main__':
    train_set, val_set, test_set = load_data('mnist.pkl.gz')
    train_x, train_y = train_set 
    val_x, val_y = val_set
    test_x, test_y = test_set
      

    #knn by the raw data
    """
    correct=0
    for k in [1,5,10]:
        for i in range(100):
            prediction=knn_algorithm(train_x[0:20000,:],test_x[i,:],train_y,k)
            if prediction==test_y[i]:
                correct+=1
        print("accuracy(raw data, k="+str(k)+") : ",(correct/100))
        correct=0
    """
    #calculating covariance of train_x.T
    #calculating the eigenvectors, eigenvalues of covariance  
    val,vec=np.linalg.eig(np.cov(train_x.T))   
    correct=0
    #knn by the projected data
   
    for dim in [2, 5, 10]:
        nvec=vec.T[0:dim]
        print(nvec.shape)
        pro_data=np.matmul(train_x,nvec.T)
        test_dim=np.matmul(test_x,nvec.T)
        
        pro_data=pro_data/(pro_data.max())
        test_dim=test_dim/(test_dim.max())
        """  
        for num in [1,5,10]:   
            for i in range(1000):
                prediction=knn_algorithm(pro_data[0:10000,:],
                                         test_dim[i,:],train_y,num)
                if(prediction==test_y[i]):
                    correct+=1
            print("accuracy(dim="+str(dim)+", k="+str(num)+") : ",correct/1000)
            correct=0
        """