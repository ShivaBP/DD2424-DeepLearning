import os
import numpy as np
import scipy.io as sc
import h5py 
import matplotlib.pyplot as plt
import matplotlib.image as img
import math
import pickle

N = 10000
d = 3072
k = 10
m = 50
mu = 0

def readData(fileName):
    path = "/Users/shivabp/Desktop/DD2424/Labs/Lab 3/Option1/cifar-10-batches-py/"
    for file in os.listdir(path):
        if (file == fileName):
            path = path + file
            #f = open(path,"rb")
            with open(path, 'rb') as f:
                dict = pickle.load(f, encoding='bytes')
                X = np.array(dict[b'data']/255)
                X = X.reshape(d,N)
                y = np.array(dict[b'labels'])
                Y = np.zeros((k, N ))
                for i in range(N):
                    Y[y[i]][i] = 1
    return X, Y , y
             
def run():
    #load data
    X, Y , y  = training = readData("data_batch_1")
    XVal , YVal , yVal = validation = readData("data_batch_2")
    XTest , YTest , yTest = readData("test_batch")

if __name__ == '__main__':
    run()
