import os
import numpy as np
import scipy.io as sc
import h5py 
import matplotlib.pyplot as plt
import matplotlib.image as img
import pickle

N = 10000
d = 3072
k = 10
sigma = 0.01
mu = 0
lamda = 0.1
eta = 0.01 
h = 1e-6
epsilon = 1e-3
n_batch = 100
n_epochs = 40

def readData(fileName):
    path = "/Users/shivabp/Desktop/DD2424/Labs/Lab 1/cifar-10-batches-py/"
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
                   
def initParams():
    W = np.random.normal(mu, sigma, (k , d))
    b = np.random.normal(mu, sigma, (k , 1))
    return W , b

def evaluateClassifier(X, W , b):
    P = np.zeros((k , X.shape[1]))
    S = np.dot(W , X) + b
    P= np.exp(S )
    denominator = np.sum(P , axis = 0)
    P = P / denominator
    return P

def computeCost(X, Y, W , b   ):
    P = evaluateClassifier(X , W , b)
    l_cross = -np.log(np.diag(np.dot(Y.T, P)))
    loss = np.sum(l_cross)
    regularization = lamda * (np.sum( np.power(W, 2))  )
    cost = (loss / X.shape[1]) + regularization
    return cost , loss

def computeAccuracy(X, y , P ): 
    predictions = np.argmax(P,axis=0)
    totalCorrect = 0
    for i in range(N):
        if( predictions[i] == y[i]  ):
            totalCorrect = totalCorrect + 1
    accuracy = (totalCorrect / N) *100
    return accuracy 

def computeGradientsNumerically(X , Y , W , b):
    grad_W = np.zeros((W.shape[0], W.shape[1]))
    grad_b = np.zeros((W.shape[0], 1))
    c , loss = computeCost(X, Y , W, b)
    for i in range (b.shape[0]):
        b_try = np.copy(b)
        b_try[i] = b_try[i] + h
        c_try1 , loss= computeCost(X, Y, W, b_try )
        grad_b[i] = ( c_try1 - c ) / h
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            W_try = W
            W_try[i][j] = W_try[i][j] +  h
            c_try2 , loss = computeCost(X, Y, W_try, b)
            grad_W[i] = (c_try2 - c)/h
    return grad_W , grad_b

def computeGradientsNumericallySlow(X , Y , W , b):
    grad_W = np.zeros((W.shape[0], W.shape[1]))
    grad_b = np.zeros((W.shape[0], 1))
    for i in range (b.shape[0]):
        b_try = np.copy(b)
        b_try[i] = b_try[i] - h
        c1 , loss = computeCost(X, Y, W, b_try )
        b_try = np.copy(b)
        b_try[i] = b_try[i] + h
        c2 , loss = computeCost(X, Y, W, b_try )
        grad_b[i] = ( c2 - c1) / (2*h)
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            W_try = W
            W_try[i][j] = W_try[i][j] -  h
            C1 , loss = computeCost(X, Y, W_try, b)
            W_try = W
            W_try[i][j]  = W_try[i][j]  +  h
            C2  , loss = computeCost(X, Y, W_try, b)
            grad_W[i][j] = (C2-C1)/(2*h)
    return grad_W , grad_b

def computeGradientsAnalytically(X , Y, W , b ):
    grad_W = np.zeros((W.shape[0] , W.shape[1]))
    grad_b = np.zeros(( k , 1))
    P = evaluateClassifier(X, W, b)
    vector = np.ones((X.shape[1] , 1 ))
    g = - (Y - P)
    grad_b = np.dot(g, vector )/ X.shape[1]  
    grad_W = np.dot( g , X.T)  / X.shape[1]
    grad_W = np.add (grad_W , (2*W))
    return grad_W, grad_b
    
def checkGradients(grad_b_Numeric , grad_w_Numeric, grad_b_Analytic , grad_w_Analytic , mode):
    grad_b_diff = np.absolute(np.subtract(grad_b_Analytic , grad_b_Numeric))
    grad_w_diff = np.absolute(np.subtract(grad_w_Analytic  , grad_w_Numeric) )
    grad_b_avg = np.mean(grad_b_diff)
    grad_w_avg = np.mean(grad_w_diff)
    if (mode == 0):      
        if(grad_b_avg  <  epsilon):
            print("gradB SUCCESS!")
            print("The average grad-b absolute difference is: " ,  grad_b_avg , "\n")   
        else:
            print("gradB FAIL!")
            print("The average grad-b absolute difference is: " ,  grad_b_avg , "\n") 
        if(grad_w_avg < epsilon):
            print("gradW SUCCESS!")
            print("The average grad-w absolute difference is: " ,  grad_w_avg , "\n")
        else:
            print("gradW FAIL!")
            print("The average grad-w absolute difference is: " ,  grad_w_avg , "\n") 
    elif(mode ==1):
        # for smaller gradient values
        grad_b_sum = np.add(np.absolute(grad_b_Analytic)  , np.absolute(grad_b_Numeric)  )
        grad_b_res = grad_b_avg / np.amax(grad_b_sum)    
        if ( grad_b_res < epsilon ):
            print("gradB SUCCESS!", "\n")
        else:
            print("gradB FAIL!", "\n")
        grad_w_sum = np.add(np.absolute(grad_w_Analytic)  , np.absolute(grad_w_Numeric)  )
        grad_w_res = grad_w_avg / np.amax(grad_w_sum) 
        if ( grad_w_res < epsilon ):
            print("gradW SUCCESS!", "\n")
        else:
            print("gradW FAIL!", "\n")
    
def miniBatchGradientDescent(X, XVal ,Y , YVal , XTest , yTest , W , b  ):
    lossValues = np.zeros(n_epochs)
    lossValValues = np.zeros(n_epochs)
    costValues = np.zeros(n_epochs)
    costValValues = np.zeros(n_epochs)
    for epoch in range(n_epochs):     
        for j in range(int ( X.shape[1] /n_batch) ):
            j_start = j*n_batch  
            j_end = j_start + n_batch
            X_batch = X[: , j_start: j_end]
            Y_batch  = Y[: , j_start:j_end]    
            wUpdate , bUpdate = computeGradientsAnalytically(X_batch , Y_batch  ,W, b)
            W = W - eta * wUpdate.reshape((k, d))
            b = b - eta * bUpdate.reshape((k, 1))
        costValues[epoch] , lossValues[epoch]  = computeCost(X, Y , W, b )
        costValValues[epoch] , lossValValues[epoch] = computeCost(XVal, YVal , W, b )     
        print("Epoch: ", epoch , "\n ")
        print("Training cost : ", costValues[epoch] , " Validation cost: " , costValValues[epoch]  ,  "\n ")
        print("Trainign loss : ", lossValues[epoch] , " Validation loss: " , lossValValues[epoch]  ,  "\n ")
    P = evaluateClassifier(XTest, W, b)
    accuracy = computeAccuracy(XTest , yTest ,P)
    print("The accuracy is: " , accuracy)
    return lossValues  , lossValValues , costValues , costValValues ,  W , b

def plotLoss(loss , lossVal , cost  , costVal):
    epochs = list(range(n_epochs))
    plt.figure(1)
    plt.plot(epochs , cost)
    plt.plot(epochs, costVal)
    plt.xlabel("Epoch number")
    plt.ylabel("Cost")
    plt.title("Cost of training across epochs")
    plt.figure(2)
    plt.plot(epochs, loss )
    plt.plot(epochs , lossVal)    
    plt.xlabel("Epoch number")
    plt.ylabel("Loss")
    plt.title("Loss of training across epochs")
    plt.show()

def plotWeights(W):
    s_im = np.zeros(k)
    for i in range(k):
        im = W[i , :].reshape(32, 32, 3)
        s_im = (im - np.amin(im) ) / (np.amax(im) - np.amin(im)  )
        plt.imshow(s_im)
        plt.show()

def run():
    # load data
    X, Y , y  = training = readData("data_batch_1")
    XVal , YVal , yVal = validation = readData("data_batch_2")
    XTest , YTest , yTest = readData("test_batch")
    # initiate patrameters
    W, b = initParams()
    #calculate, loss, weights and cost and plot them
    '''
    lossValues  , lossValValues , costValues , costValValues ,  W , b = miniBatchGradientDescent(X, XVal ,  Y, YVal , XTest , yTest , W , b)   
    plotWeights(W)
    plotLoss(lossValues , lossValValues , costValues , costValValues)
    '''
    #Check gradients reliability
    grad_w , grad_b = computeGradientsAnalytically(X[:5 , 0:2], Y[: , 0:2] , W[: , :5] , b )
    grad_w1 , grad_b1 = computeGradientsNumerically(X[:5 , 0:2], Y[: , 0:2] , W[: , :5] , b )
    grad_w2 , grad_b2 = computeGradientsNumericallySlow(X[:5 , 0:2], Y[: , 0:2] , W[: , :5] , b)
    checkGradients(grad_b2 , grad_w2, grad_b , grad_w , 0)
    checkGradients(grad_b1 , grad_w1, grad_b , grad_w , 0)  
    checkGradients(grad_b2 , grad_w2, grad_b , grad_w , 1)
    checkGradients(grad_b1 , grad_w1, grad_b , grad_w , 1)   
    
if __name__ == '__main__':
    run()
    '''
    To do:
    - Fix gradient check gradW  
    - fix accuracy 
    - report
    '''