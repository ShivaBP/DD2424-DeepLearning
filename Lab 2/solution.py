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
sigma1 =  1 / math.sqrt(d)
sigma2 = 1 / math.sqrt(m)
lamda = 0.01
eta = 1e-4
eta_min = 1e-5
eta_max = 1e-1
n_s = 500
epsilon = 1e-4
h = 1e-5
n_batch = 100
n_epochs = 10

def readData(fileName):
    path = "/Users/shivabp/Desktop/DD2424/Labs/Lab 2/cifar-10-batches-py/"
    for file in os.listdir(path):
        if (file == fileName):
            path = path + file
            #f = open(path,"rb")
            with open(path, 'rb') as f:
                dict = pickle.load(f, encoding='bytes')
                X = np.array(dict[b'data'])
                X = X.reshape(d,N)
                y = np.array(dict[b'labels'])
                Y = np.zeros((k, N ))
                for i in range(N):
                    Y[y[i]][i] = 1
    return X, Y , y

def repmat(array , dim1 , dim2):
    matrix = np.matlib.repmat(array , dim1 , dim2)
    return matrix 

def normalize(X , mean_X , std_X ):
    meanRepmat = repmat(mean_X , 1, X.shape[1])
    stdRepmat = repmat(std_X , 1 , X.shape[1])
    X = X.reshape(1 , X.shape[0]*X.shape[1])
    X = np.subtract (X , meanRepmat )
    X = np.divide(X, stdRepmat)
    X = X.reshape(d , N )
    return X

def initParams():
    W1 = np.random.normal(mu, sigma1, (m , d))
    W2 = np.random.normal(mu, sigma2, ( k ,  m))
    b1 = np.zeros((m , 1))
    b2 = np.zeros((k , 1))
    return W1, W2 , b1 , b2

def evaluateClassifier(X , W1, W2 , b1 , b2):
    S1 = np.add (np.dot(W1 , X) , b1 )
    h = np.maximum(0 , S1)
    S = np.add (np.dot(W2 , h) , b2 ) 
    P = np.exp(S  )
    denominator = np.sum(P , axis =0) 
    P = P  / denominator
    return h, P

def computeCost(X, Y , W1 , W2 , b1, b2):
    h, P = evaluateClassifier(X , W1, W2 , b1 , b2)
    l_cross = -np.log(np.diag(np.dot(Y.T, P)))
    loss = np.sum(l_cross)
    regularization = lamda * np.add(np.sum( np.power(W1 , 2)) , np.sum(np.power(W2 , 2)) )
    cost = (loss / X.shape[1]) + regularization
    return loss, cost

def computeAccuracy(X, y , W1 , W2 , b1  , b2 ): 
    h, P = evaluateClassifier(X, W1, W2 , b1 , b2)
    predictions = np.argmax(P,axis=0)
    totalCorrect = 0
    for i in range(X.shape[1]):
        if( predictions[i] ==  y[i] ):
            totalCorrect = totalCorrect + 1
    accuracy = (totalCorrect / N) *100
    return accuracy 

def computeGradientsAnalytically(X , Y, W1 , W2 , b1, b2 ):
    # lecture 4, slides 30-33
    grad_W1 = np.zeros((W1.shape[0] , W1.shape[1]))
    grad_W2 = np.zeros((W2.shape[0] , W2.shape[1]))
    grad_b1 = np.zeros(( m , 1))
    grad_b2 = np.zeros(( k , 1))
    h , P = evaluateClassifier(X, W1, W2 , b1, b2)
    indicator = 1 * (h > 0)
    vector = np.ones((X.shape[1] , 1))
    g = - (Y - P)  
    grad_b2 =  np.dot(g , vector)/ X.shape[1] 
    grad_W2 = np.dot( g , h.T). reshape(W2.shape[0] , W2.shape[1])/X.shape[1]
    g = np.dot(W2.T , g)
    g = np.multiply(g, indicator)
    grad_b1 = np.dot(g, vector).reshape((m, 1))/ X.shape[1] 
    grad_W1 =  np.dot( g , X.T).reshape((W1.shape[0] , W1.shape[1]))/ X.shape[1]  
    grad_W1 = np.add (grad_W1 , 2*lamda*W1)
    grad_W2 = np.add( grad_W2 , 2*lamda*W2)
    return grad_b1 , grad_b2 , grad_W1 , grad_W2
   
def computeGradsNumerically(X, Y, W1, W2 , b1, b2):
    grad_W1 = np.zeros((W1.shape[0] , W1.shape[1]))
    grad_W2 = np.zeros((W2.shape[0] , W2.shape[1]))
    grad_b1 = np.zeros(( m , 1))
    grad_b2 = np.zeros(( k , 1))
    loss , cost = computeCost(X, Y , W1 , W2 , b1, b2)
    for i in range(len(b1)):
        b1_try = np.copy(b1)
        b1_try[i] = b1_try[i] + h
        loss , c_try1 = computeCost(X, Y, W1, W2, b1_try , b2)
        grad_b1[i] = (c_try1 - cost) / h
    for i in range(len(b2)):
        b2_try = np.copy(b2)
        b2_try[i] = b2_try[i] + h
        loss , c_try2 = computeCost(X, Y, W1, W2, b1 , b2_try)
        grad_b2[i] = (c_try2 - cost) / h
    for i in range(W1.shape[0]):
        for j in range(W1.shape[1]):
            W_try1 = np.copy(W1)
            W_try1[i][j] = W_try1[i][j] + h 
            loss , C_try1 = computeCost(X, Y, W_try1, W2, b1 , b2)
            grad_W1[i][j] = (C_try1 - cost) / h
    for i in range(W2.shape[0]):
        for j in range(W2.shape[1]):
            W_try2 = np.copy(W2)
            W_try2[i][j] = W_try2[i][j] + h 
            loss , C_try2 = computeCost(X, Y, W1, W_try2, b1 , b2)
            grad_W2[i][j] = (C_try2 - cost) / h
    return grad_b1 , grad_b2 , grad_W1 , grad_W2

def computeGradsNumericallySlow(X, Y, W1, W2 , b1, b2):
    grad_W1 = np.zeros((W1.shape[0] , W1.shape[1]))
    grad_W2 = np.zeros((W2.shape[0] , W2.shape[1]))
    grad_b1 = np.zeros(( m , 1))
    grad_b2 = np.zeros(( k , 1))
    for i in range(len(b1)):
        b1_try = np.copy(b1)
        b1_try[i] = b1_try[i] - h
        loss , c_try11 = computeCost(X, Y, W1, W2, b1_try , b2)
        b1_try = np.copy(b1)
        b1_try[i] = b1_try[i] +  h
        loss , c_try21 = computeCost(X, Y, W1, W2, b1_try , b2)
        grad_b1[i] = (c_try21 - c_try11) / (2*h)
    for i in range(len(b2)):
        b2_try = np.copy(b2)
        b2_try[i] = b2_try[i] -  h
        loss , c_try12 = computeCost(X, Y, W1, W2, b1 , b2_try)
        b2_try = np.copy(b2)
        b2_try[i] = b2_try[i] +  h
        loss , c_try22 = computeCost(X, Y, W1, W2, b1 , b2_try)
        grad_b2[i] = (c_try22 - c_try12) / (2*h)
    for i in range(W1.shape[0]):
        for j in range(W1.shape[1]):
            W_try1 = np.copy(W1)
            W_try1[i][j] = W_try1[i][j] - h 
            loss , C_try11 = computeCost(X, Y, W_try1, W2, b1 , b2)
            W_try1 = np.copy(W1)
            W_try1[i][j] = W_try1[i][j] + h 
            loss , C_try21 = computeCost(X, Y, W_try1, W2, b1 , b2)
            grad_W1[i][j] = (C_try21 - C_try11 ) / (2*h)
    for i in range(W2.shape[0]):
        for j in range(W2.shape[1]):
            W_try2 = np.copy(W2)
            W_try2[i][j] = W_try2[i][j] - h 
            loss , C_try12 = computeCost(X, Y, W1, W_try2, b1 , b2)
            W_try2 = np.copy(W2)
            W_try2[i][j] = W_try2[i][j] + h 
            loss , C_try22 = computeCost(X, Y, W1, W_try2, b1 , b2)
            grad_W2[i][j] = (C_try22 - C_try12) / (2*h)
    return grad_b1 , grad_b2 , grad_W1 , grad_W2

def checkGradients(grad_b_Numeric , grad_w_Numeric, grad_b_Analytic , grad_w_Analytic , mode, layer):
    print("Results for layer number ", layer, " in mode ", mode, "\n")
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
            print("gradW SUCCESS!!", "\n")
        else:
            print("gradW SUCCESS!!", "\n")

def miniBatchGradientDescent(X, XVal ,Y , YVal , XTest , yTest , W1, W2  , b1, b2  ):
    lossValues = np.zeros(n_epochs)
    lossValValues = np.zeros(n_epochs)
    costValues = np.zeros(n_epochs)
    costValValues = np.zeros(n_epochs)
    tempW1 = np.zeros(np.shape(W1))
    tempW2 = np.zeros(np.shape(W2))
    tempb1 = np.zeros(np.shape(b1))
    tempb2 = np.zeros(np.shape(b2))  
    for epoch in range(n_epochs):     
        for j in range(int ( X.shape[1] /n_batch) ):
            j_start = j*n_batch  
            j_end = j_start + n_batch
            X_batch = X[: , j_start: j_end]
            Y_batch  = Y[: , j_start:j_end]    
            grad_b1 , grad_b2 , grad_W1 , grad_W2 = computeGradientsAnalytically(X_batch , Y_batch  ,W1,W2, b1 , b2)
            tempW1 = tempW1 - eta * grad_W1.reshape((m, d))
            W1 = W1 + tempW1
            tempb1 = tempb1 - eta * grad_b1.reshape((m, 1))
            b1 = b1 + tempb1
            tempW2 = tempW2 - eta * grad_W2.reshape((k, m))
            W2 = W2 + tempW2   
            tempb2 = tempb2  - eta * grad_b2.reshape((k, 1))
            b2 = b2 + tempb2       
        lossValues[epoch] , costValues[epoch]  = computeCost(X, Y , W1 , W2 , b1 , b2 )
        lossValValues[epoch] , costValValues[epoch] = computeCost(X, Y , W1 , W2 , b1 , b2 )     
        print("Epoch: ", epoch , "\n ")
        print("Training cost : ", costValues[epoch] , " Validation cost: " , costValValues[epoch]  ,  "\n ")
        print("Trainign loss : ", lossValues[epoch] , " Validation loss: " , lossValValues[epoch]  ,  "\n ")
    accuracy = computeAccuracy(XTest , yTest , W1 , W2 , b1 , b2)
    print("The final accuracy is: " , accuracy)
    return lossValues  , lossValValues , costValues , costValValues ,  W1 , W2 ,  b1 , b2

def run():
    #load data
    X, Y , y  = training = readData("data_batch_1")
    XVal , YVal , yVal = validation = readData("data_batch_2")
    XTest , YTest , yTest = readData("test_batch")
    # Normalization
    mean_X = np.mean(X, axis = 1)
    std_X = np.std(X, axis= 1)
    X = normalize(X, mean_X , std_X)
    XVal = normalize(XVal, mean_X , std_X)
    XTest = normalize(XTest, mean_X , std_X)
    #parameters 
    W1, W2 , b1, b2 = initParams()  
    #check gradients, first check grad_b1 and grad_W1 in both modes and then grad_b2, gradW2
    grad_b1 , grad_b2 , grad_W1 , grad_W2 = computeGradientsAnalytically(X[:5 , 0:2], Y[: , 0:2] , W1[: , :5], W2 , b1 , b2)
    grad_b11 , grad_b21 , grad_W11 , grad_W21 = computeGradsNumerically(X[:5 , 0:2], Y[: , 0:2] , W1[: , :5], W2 , b1 , b2)
    grad_b12 , grad_b22 , grad_W12 , grad_W22 = computeGradsNumericallySlow(X[:5 , 0:2], Y[: , 0:2] , W1[: , :5], W2 , b1 , b2)
    print("Comparison between analytically computed gradients and fast numerically computed gradients:" , "\n")
    checkGradients(grad_b11 , grad_W11, grad_b1 , grad_W1 , 0 , 1)
    checkGradients(grad_b11 , grad_W11, grad_b1 , grad_W1 , 1 , 1)
    checkGradients(grad_b21 , grad_W21, grad_b2 , grad_W2 , 0, 2)
    checkGradients(grad_b21 , grad_W21, grad_b2 , grad_W2 , 1, 2)
    print("Comparison between analytically computed gradients and slow numerically computed gradients:" , "\n")
    checkGradients(grad_b12 , grad_W12, grad_b1 , grad_W1 , 0, 1)
    checkGradients(grad_b12 , grad_W12, grad_b1 , grad_W1 , 1, 1)    
    checkGradients(grad_b22 , grad_W22, grad_b2 , grad_W2 , 0, 2)
    checkGradients(grad_b22 , grad_W22, grad_b2 , grad_W2 , 1, 2)
    '''
    # minibatch gradient descent
    #lossValues,lossValValues,costValues,costValValues,W1,W2,b1,b2 =miniBatchGradientDescent(X[: , :10], XVal[: , :10],Y[: , :10], YVal[: , :10], XTest[: , :10], yTest[:10] , W1, W2, b1, b2 )
    lossValues,lossValValues,costValues,costValValues,W1,W2,b1,b2 =miniBatchGradientDescent(X, XVal,Y, YVal, XTest, yTest , W1, W2, b1, b2 )
    '''
if __name__ == '__main__':
    run()
    '''
    To do: 
    - fix eta
    - fix accuracy 
    - fix plots 
    - report
    '''