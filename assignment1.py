import os
import numpy as np
import scipy.io as sc
import h5py 

N = 10000
d = 3072
k = 10
sigma = 0.01
mu = 0
lamda = 0
eta = 0.01 
h = 1e-6
epsilon = 1e-6
n_batch = 100 
n_epochs = 40

def readData(fileName):
    path = "/Users/shivabp/Desktop/DD2424/Labs/Lab 1/cifar-10-batches-mat/"
    for file in os.listdir(path):
        if (file == fileName):
            path = path + file
            f = sc.loadmat(path)
            X = np.array(f['data'])/ 255       
            #X = np.swapaxes(X,0,1)
            X = X.reshape(d,N)
            y= np.array(f['labels']).reshape(N)
            Y = np.zeros((k, N ))
            for i in range(N):
                Y[y[i]][i] = 1
    return X, Y , y
                    
def initParams():
    W = np.random.normal(mu, sigma, (k , d))
    b = np.random.normal(mu, sigma, (k , 1))
    return W , b

def evaluateClassifier(X, W , b):
    S = np.dot(W, X) + b  
    P = np.exp(S)
    denominator = np.sum(P , axis=0)
    P = P / denominator
    return P

def computeLoss(X, Y, W , b  ):
    P = evaluateClassifier(X, W , b)
    crossEntropyLoss =   - np.log (np.diag(np.dot ( Y.T , P )  ) )
    loss = np.sum(crossEntropyLoss )
    return loss

def computeCost(X, Y, W , b   ):
    regularization = lamda * np.sum(np.square(W))
    loss = computeLoss(X, Y, W, b)
    cost = loss/X.shape[1]  + regularization 
    return cost 

def computeAccuracy(X, y , W , b  ): 
    P = evaluateClassifier(X, W, b) 
    predictions = np.argmax(P, axis=0)
    totalCorrect = 0 
    for i in range(X.shape[1]):
        if (predictions[i] == y[i]):
            totalCorrect = totalCorrect + 1
    accuracy = (totalCorrect / X.shape[1]) * 100
    return accuracy

def computeGradientsNumerically(X , Y , W , b):
    grad_W = np.zeros((W.shape[0], W.shape[1]))
    grad_b = np.zeros((W.shape[0], 1))
    c = computeCost(X, Y , W, b)
    for i in range (b.shape[0]):
        b_try = np.copy(b)
        b_try[i] = b_try[i] + h
        c_try = computeCost(X, Y, W, b_try )
        grad_b[i] = ( c_try - c) / h
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            W_try = W
            W_try[i][j] = W_try[i][j] +  h
            c_try = computeCost(X, Y, W_try, b)
            grad_W[i][j] = (c_try-c)/h
    return grad_W , grad_b

def computeGradientsNumericallySlow(X , Y , W , b):
    grad_W = np.zeros((W.shape[0], W.shape[1]))
    grad_b = np.zeros((W.shape[0], 1))
    for i in range (b.shape[0]):
        b_try = np.copy(b)
        b_try[i] = b_try[i] - h
        c1 = computeCost(X, Y, W, b_try )
        b_try = np.copy(b)
        b_try[i] = b_try[i] + h
        c2 = computeCost(X, Y, W, b_try )
        grad_b[i] = ( c2 - c1) / (2*h)
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            W_try = W
            W_try[i][j] = W_try[i][j] -  h
            c1 = computeCost(X, Y, W_try, b)
            W_try = W
            W_try[i][j]  = W_try[i][j]  +  h
            c2 = computeCost(X, Y, W_try, b)
            grad_W[i][j] = (c2-c1)/(2*h)
    return grad_W , grad_b

def computeGradientsAnalytically(X , Y, P , W , b ):
    # Refer to lecture 3 
    g = - (Y - P).T
    grad_b = np.mean(g.T , 1)
    grad_b = np.reshape(grad_b, (-1, 1))  
    grad_W = (np.dot(g.T, X.T))/X.shape[1] + 2*lamda * W  
    return grad_W, grad_b

def checkGradients(grad_b_Numeric , grad_w_Numeric, grad_b_Analytic , grad_w_Analytic , mode):
    grad_b_diff = np.absolute(grad_b_Analytic - grad_b_Numeric) 
    grad_w_diff = np.absolute(grad_w_Analytic - grad_w_Numeric) 
    grad_b_avg = np.average(grad_b_diff)
    grad_w_avg = np.average(grad_w_diff)
    if (mode == 0):      
        print("The average grad-b absolute difference is: " ,  grad_b_avg , "\n")     
        print("The average grad-w absolute difference is: " ,  grad_w_avg , "\n")
        if ( grad_b_avg < epsilon and  grad_w_avg < epsilon):
            print("The results are reliable!", "\n")
        else:
            print("The results are not reliable!", "\n")
    elif(mode ==1):
        # for smaller gradient values
        grad_b_sum = np.sum(np.absolute(grad_b_Analytic)  , np.absolute(grad_b_Numeric)  )
        grad_b_res = grad_b_avg / np.amax(grad_b_sum)    
        grad_w_sum = np.sum(np.absolute(grad_b_Analytic)  , np.absolute(grad_b_Numeric)  )
        grad_w_res = grad_w_avg / np.amax(grad_w_sum) 
        if ( grad_b_res < epsilon  and grad_w_res < epsilon):
            print("The results are reliable!", "\n")
        else:
            print("The results are not reliable!", "\n")

def miniBatchGradientDescent(X, XVal ,Y , YVal , y , yVal , W , b  ):
    Wstar = W
    bstar = b
    lossValues = np.zeros(n_epochs)
    lossValValues = np.zeros(n_epochs)
    costValues = np.zeros(n_epochs)
    costValValues = np.zeros(n_epochs)
    for epoch in range(n_epochs):
        print("Epoch: ", epoch , "\n ")
        for j in range(int ( X.shape[1] /n_batch) ):
            j_start = j*n_batch 
            j_end = (j+ 1) *n_batch
            X_batch = X[: , j_start: j_end]
            Y_batch  = Y[: , j_start:j_end]
            P = evaluateClassifier(X_batch , Wstar , bstar)     
            wUpdate , bUpdate = computeGradientsAnalytically(X_batch , Y_batch , P ,Wstar, bstar)
            Wstar= Wstar - eta*wUpdate
            bstar = bstar - eta*bUpdate
        costValues[epoch] = computeCost(X, Y , Wstar, bstar )
        costValValues[epoch] = computeCost(XVal, YVal , Wstar, bstar )
        print("Training cost : ", costValues[epoch] , " Validation cost: " , costValValues[epoch]  ,  "\n ")
        lossValues[epoch] = computeLoss(X, Y, Wstar, bstar)
        lossValValues[epoch] = computeLoss(XVal, YVal, Wstar, bstar)
        print("Trainign loss : ", costValues[epoch] , " Validation loss: " , costValValues[epoch]  ,  "\n ")
    return lossValues  , lossValValues , costValues , costValValues ,  Wstar , bstar

def plot(loss , cost  , inputMode):
    if (inputMode == 0):
        figure1, ax = plt.subplots(2, sharex=True)
        ax[0].plot(loss,label = 'Training set')
        ax[1].plot(cost,label = 'Training set')
        figure2 =plt.figure()
        plt.plot(accuracy,label = 'Training set')
        plt.legend()
        plt.show()
    elif(inputMode == 1):
        figure1, ax = plt.subplots(2, sharex=True)
        ax[0].plot(loss,label = 'Validation set')
        ax[1].plot(cost,label = 'Validation set')
        figure2 =plt.figure()
        plt.plot(accuracy,label = 'Validation set')
        plt.legend()
        plt.show()
    elif(inputMode == 2):
        figure1, ax = plt.subplots(2, sharex=True)
        ax[0].plot(loss,label = 'Test set')
        ax[1].plot(cost,label = 'Test set')
        figure2 =plt.figure()
        plt.plot(accuracy,label = 'Test set')
        plt.legend()
        plt.show()

def run():
    X, Y , y  = training = readData("data_batch_1.mat")
    XVal , YVal , yVal = validation = readData("data_batch_2.mat")
    XTest , YTest , yTest = readData("test_batch.mat")
    W, b = initParams()

    lossValues  , lossValValues , costValues , costValValues ,  Wstar , bstar = miniBatchGradientDescent(X, XVal ,  Y, YVal ,  y , yVal , W , b)
    accuracy = computeAccuracy(XTest , yTest , Wstar , bstar)
    print("The accuracy is: " , accuracy)
  
  '''
  To do:
  - Fix gradient check 
  - fix plots 
  - Report 
  '''

run()