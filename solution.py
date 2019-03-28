import os
import numpy as np
import scipy.io as sc
import h5py 
import matplotlib.pyplot as plt
import matplotlib.image as img

N = 10000
d = 3072
k = 10
sigma = 0.01
mu = 0
lamda = 1
eta = 0.01 
h = 1e-6
epsilon = 1e-6
n_batch = 1
n_epochs = 2

def readData(fileName):
    path = "/Users/shivabp/Desktop/DD2424/Labs/Lab 1/cifar-10-batches-mat/"
    for file in os.listdir(path):
        if (file == fileName):
            path = path + file
            f = sc.loadmat(path)
            X = np.array(f['data'])/ 255       
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
    predictions = np.argmax(P,axis=0)
    totalCorrect = 0
    for i in range(N):
        if( predictions[i] -  y[i] == 0 ):
            totalCorrect = totalCorrect + 1
    accuracy = (totalCorrect / N) *100
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
            grad_W[i][j] = (c_try - c)/h
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

def computeGradientsAnalytically(X , Y, W , b ):
    grad_W = np.zeros((W.shape[0] , W.shape[1]))
    grad_b = np.zeros(( k , 1))
    P = evaluateClassifier(X, W, b)
    for i in range(X.shape[1]):
        Yi = Y[:, i] .reshape((Y[:, i].shape[0], 1))
        Pi = P[:, i] .reshape((P[:, i].shape[0], 1))
        Xi = X[:, i] .reshape((X[:, i].shape[0] , 1))    
        g = - (Yi - Pi)
        grad_b = grad_b +  g     
        grad_W = grad_W + np.dot ( g , Xi.T)   
    grad_b = np.divide(grad_b, X.shape[1])
    grad_W = np.divide(grad_W, X.shape[1])
    grad_W = grad_W + (2 * lamda * W) 
    return grad_W, grad_b
    
def checkGradients(grad_b_Numeric , grad_w_Numeric, grad_b_Analytic , grad_w_Analytic , mode):
    grad_b_diff = np.absolute(np.subtract(grad_b_Analytic , grad_b_Numeric))
    grad_w_diff = np.absolute(np.subtract(grad_w_Analytic  , grad_w_Numeric) )
    grad_b_avg = np.mean(grad_b_diff)
    grad_w_avg = np.mean(grad_w_diff)
    if (mode == 0):      
        print("The average grad-b absolute difference is: " ,  grad_b_avg , "\n")     
        print("The average grad-w absolute difference is: " ,  grad_w_avg , "\n")
        if ( grad_b_avg < epsilon and  grad_w_avg < epsilon):
            print("The results are reliable!", "\n")
        else:
            print("The results are not reliable!", "\n")
    elif(mode ==1):
        # for smaller gradient values
        grad_b_sum = np.add(np.absolute(grad_b_Analytic)  , np.absolute(grad_b_Numeric)  )
        grad_b_res = grad_b_avg / np.amax(grad_b_sum)    
        grad_w_sum = np.add(np.absolute(grad_b_Analytic)  , np.absolute(grad_b_Numeric)  )
        grad_w_res = grad_w_avg / np.amax(grad_w_sum) 
        if ( grad_b_res < epsilon  and grad_w_res < epsilon):
            print("The results are reliable!", "\n")
        else:
            print("The results are not reliable!", "\n")
    
def miniBatchGradientDescent(X, XVal ,Y , YVal , XTest , yTest , W , b  ):
    Wstar = W
    bstar = b
    accuracy = computeAccuracy(XTest , yTest , Wstar , bstar)
    print("The accuracy is: " , accuracy)
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
            wUpdate , bUpdate = computeGradientsAnalytically(X_batch , Y_batch  ,Wstar, bstar)
            Wstar -=  eta * wUpdate  
            bstar -=  eta * bUpdate   
        costValues[epoch] = computeCost(X, Y , Wstar, bstar )
        costValValues[epoch] = computeCost(XVal, YVal , Wstar, bstar )     
        lossValues[epoch] = computeLoss(X, Y, Wstar, bstar)
        lossValValues[epoch] = computeLoss(XVal, YVal, Wstar, bstar)
        print("Epoch: ", epoch , "\n ")
        print("Training cost : ", costValues[epoch] , " Validation cost: " , costValValues[epoch]  ,  "\n ")
        print("Trainign loss : ", lossValues[epoch] , " Validation loss: " , lossValValues[epoch]  ,  "\n ")
    accuracy = computeAccuracy(XTest , yTest , Wstar , bstar)
    print("The accuracy is: " , accuracy)
    return lossValues  , lossValValues , costValues , costValValues ,  Wstar , bstar

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
    X, Y , y  = training = readData("data_batch_1.mat")
    XVal , YVal , yVal = validation = readData("data_batch_2.mat")
    XTest , YTest , yTest = readData("test_batch.mat")

    W, b = initParams()
    '''
    lossValues  , lossValValues , costValues , costValValues ,  Wstar , bstar = miniBatchGradientDescent(X, XVal ,  Y, YVal , XTest , yTest , W , b)
    
    plotWeights(Wstar)
    
    plotLoss(lossValues , lossValValues , costValues , costValValues)
    '''
    [grad_w , grad_b] = computeGradientsAnalytically(X[:20 , 0:2], Y[: , 0:2] , W[: , :20] , b )
    [grad_w1 , grad_b1] = computeGradientsNumerically(X[:20 , 0:2], Y[: , 0:2] , W[: , :20] , b )
    [grad_w2 , grad_b2] = computeGradientsNumericallySlow(X[:20 , 0:2], Y[: , 0:2] , W[: , :20] , b)
    checkGradients(grad_b1 , grad_w1, grad_b , grad_w , 1)
    checkGradients(grad_b1 , grad_w1, grad_b , grad_w , 0)  
    checkGradients(grad_b2 , grad_w2, grad_b , grad_w , 1)
    checkGradients(grad_b2 , grad_w2, grad_b , grad_w , 0)
    
if __name__ == '__main__':
    run()
    '''
    To do:
    - Fix gradient check  
    - fix accuracy ( around 10%)
    - report
    '''