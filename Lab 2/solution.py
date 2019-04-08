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
h = 1e-5
eta_min = 1e-5
eta_max = 1e-1
n_batch = 100
n_epochs = 10
iters = int (n_epochs*(N/n_batch))
n_s = 500
cycle = 1
lamda = 0.01

def readData(fileName):
    path = "/Users/shivabp/Desktop/DD2424/Labs/Lab 1/cifar-10-batches-py/" + fileName
    with open(path, 'rb') as f:
        data = pickle.load(f, encoding='bytes')
    f.close()
    X = np.array(data[b'data']/255).T
    y = np.array(data[b'labels'])
    Y = np.zeros((k, N ))
    for i in range(N):
        Y[y[i]][i] = 1
    return X, Y, y

def normalize(trainX , X ):
    mean_X = np.mean(trainX)
    std_X = np.std(trainX)
    X = np.subtract (X , mean_X) / std_X
    return X

def initParams():
    W1 = np.random.normal(mu, sigma1, (m , d))
    W2 = np.random.normal(mu, sigma2, ( k ,  m))
    b1 = np.zeros((m , 1))
    b2 = np.zeros((k , 1))
    return W1, W2 , b1 , b2

def cycleETA(iteration):
    difference = eta_max - eta_min
    x = abs((iteration/ n_s) - (2*cycle) +1)
    eta = eta_min + (difference * np.maximum(0, (1-x)))
    return eta

def evaluateClassifier(X , W1, W2 , b1 , b2):
    S1 = np.dot(W1 , X) + b1 
    activations = np.maximum(0 , S1)
    S = np.dot(W2 , activations) + b2  
    numerator = np.exp(S  )
    probabilities = numerator  / np.sum(numerator , axis =0) 
    predictions = np.argmax(probabilities, axis=0)
    return activations, probabilities , predictions

def computeCost(probabilities,Y ,  W1 , W2 ):
    py = np.multiply(Y, probabilities).sum(axis=0)
    # avoid the error
    py [py  == 0] = np.finfo(float).eps
    l2Reg = lamda * (np.sum( np.square(W1)) + np.sum(np.square(W2)) )
    return ((-np.log(py).sum() / probabilities.shape[1] )  + l2Reg )

def computeAccuracy(predictions, y ): 
    totalCorrect = 0
    for i in range(predictions.shape[0]):
        if( predictions[i] ==  y[i] ):
            totalCorrect = totalCorrect + 1
    accuracy = (totalCorrect / predictions.shape[0]) *100
    return accuracy 

def computeGradAnalytic(X , Y, W1 , W2 , b1, b2 ):
    # lecture 4, slides 30-33
    grad_W1 = np.zeros((W1.shape[0] , W1.shape[1]))
    grad_W2 = np.zeros((W2.shape[0] , W2.shape[1]))
    grad_b1 = np.zeros(( m , 1))
    grad_b2 = np.zeros(( k , 1))
    activations , probabilities , predictions = evaluateClassifier(X, W1, W2 , b1, b2)
    indicator = 1 * (activations > 0)
    vector = np.ones((X.shape[1] , 1))
    g = - (Y - probabilities)  
    grad_b2 =  np.dot(g , vector)/ X.shape[1]  
    grad_W2 = np.dot( g , activations.T)/X.shape[1]
    g = np.dot(W2.T , g)
    g = np.multiply(g, indicator)
    grad_b1 = np.dot(g, vector)/ X.shape[1] 
    grad_W1 =  np.dot( g , X.T)/ X.shape[1]  
    grad_W1 = grad_W1 + (2*lamda*W1)
    grad_W2 = grad_W2 + (2*lamda*W2)
    return grad_b1 , grad_b2 , grad_W1 , grad_W2
   
def computeGradNumeric(X, Y, W1, W2 , b1, b2):
    grad_W1 = np.zeros((W1.shape[0] , W1.shape[1]))
    grad_W2 = np.zeros((W2.shape[0] , W2.shape[1]))
    grad_b1 = np.zeros(( m , 1))
    grad_b2 = np.zeros(( k , 1))
    activations, probabilities, predictions = evaluateClassifier(X, W1 , W2 , b1, b2)
    cost = computeCost(probabilities, Y ,  W1, W2)
    for i in range(b1.shape[0]):
        b1[i] += h
        activations, probabilities, predictions = evaluateClassifier(X, W1 , W2 , b1, b2)
        cost_try = computeCost(probabilities, Y, W1, W2)
        grad_b1[i] = (cost_try - cost) / h
        b1[i] -= h
    for i in range(b2.shape[0]):
        b2[i] += h
        activations, probabilities, predictions = evaluateClassifier(X, W1 , W2 , b1, b2)
        cost_try = computeCost(probabilities, Y, W1, W2)
        grad_b2[i] = (cost_try - cost) / h
        b2[i] -= h
    for i in range(W1.shape[0]):
        for j in range(W1.shape[1]):
            W1[i][j] += h
            activations, probabilities, predictions = evaluateClassifier(X, W1 , W2 , b1, b2)
            cost_try = computeCost(probabilities, Y, W1, W2)
            grad_W1[i, j] = (cost_try - cost) / h
            W1[i][j] -= h
    for i in range(W2.shape[0]):
        for j in range(W2.shape[1]):
            W2[i][j] += h
            activations, probabilities, predictions = evaluateClassifier(X, W1 , W2 , b1, b2)
            cost_try = computeCost(probabilities, Y, W1, W2)
            grad_W2[i, j] = (cost_try - cost) / h
            W2[i][j] -= h
    return grad_b1 , grad_b2 , grad_W1 , grad_W2

def checkGradients():
    X, Y, y = readData("data_batch_1")
    X = normalize(X, X)
    W1, W2 , b1 , b2 = initParams()
    grad_b1Analytic , grad_b2Analytic , grad_W1Analytic , grad_W2Analytic  = computeGradAnalytic(X[:20 , 0:1], Y[: , 0:1] , W1[: , :20] , W2 , b1, b2 )
    grad_b1Numeric , grad_b2Numeric , grad_W1Numeric , grad_W2Numeric = computeGradNumeric(X[:20 , 0:1], Y[: , 0:1] , W1[: , :20] , W2 , b1, b2 )
    print("gradW1 results:" )
    print('Average of absolute differences is: ' , np.mean (np.abs(grad_W1Analytic - grad_W1Numeric)) )
    print("Analytic gradW1:  Mean:   " ,np.abs(grad_W1Analytic).mean() , "   Min:    " ,np.abs(grad_W1Analytic).min() , "    Max:    " ,  np.abs(grad_W1Analytic).max())
    print("Numeric gradW1:   Mean:   " ,np.abs(grad_W1Numeric).mean() ,  "   Min:    " ,np.abs(grad_W1Numeric).min() ,  "    Max:    " ,  np.abs(grad_W1Numeric).max(), "\n")
    print("gradW2 results:" )
    print('Average of absolute differences is: ' , np.mean (np.abs(grad_W2Analytic - grad_W2Numeric)))
    print("Analytic gradW2:  Mean:   " ,np.abs(grad_W2Analytic).mean() , "   Min:    " ,np.abs(grad_W2Analytic).min() , "    Max:    " ,  np.abs(grad_W2Analytic).max())
    print("Numeric gradW2:   Mean:   " ,np.abs(grad_W2Numeric).mean() ,  "   Min:    " ,np.abs(grad_W2Numeric).min() ,  "    Max:    " ,  np.abs(grad_W2Numeric).max(), "\n")
    print("gradB1 results:" )
    print('Average of absolute differences is: ' ,np.mean ( np.abs(grad_b1Analytic - grad_b1Numeric) ) )
    print("Analytic gradb1:  Mean:   " ,np.abs(grad_b1Analytic).mean() , "   Min:    " ,np.abs(grad_b1Analytic).min() , "    Max:    " ,  np.abs(grad_b1Analytic).max())
    print("Numeric gradb1:   Mean:   " ,np.abs(grad_b1Numeric).mean() ,  "   Min:    " ,np.abs(grad_b1Numeric).min() ,  "    Max:    " ,  np.abs(grad_b1Numeric).max(), "\n")
    print("gradB2 results:" )
    print('Average of absolute differences is: ' , np.mean (np.abs(grad_b2Analytic - grad_b2Numeric)))
    print("Analytic gradb2:  Mean:   " ,np.abs(grad_b2Analytic).mean() , "   Min:    " ,np.abs(grad_b2Analytic).min() , "    Max:    " ,  np.abs(grad_b2Analytic).max())
    print("Numeric gradb2:   Mean:   " ,np.abs(grad_b2Numeric).mean() ,  "   Min:    " ,np.abs(grad_b2Numeric).min() ,  "    Max:    " ,  np.abs(grad_b2Numeric).max(), "\n")

def miniBatchGradientDescent( W1, W2  , b1, b2 ):
    # load data
    X, Y , y  = training = readData("data_batch_1")
    XVal , YVal , yVal = validation = readData("data_batch_2")
    XTest , YTest , yTest = readData("test_batch")
    # Normalization
    X = normalize(X, X)
    XVal = normalize(X ,XVal)
    XTest = normalize(X , XTest)
    '''
    #Used for testing
    X = X[: , :100]
    Y = Y[: , :100]
    XVal  = XVal[: , :100]
    YVal = YVal[: , :100]
    XTest =  XTest[: , :100]
    yTest = yTest[:100]
    '''
    #Store results 
    accuracyValues = list()
    accuracyValValues = list()
    costValues = list()
    costValValues = list()
    iter = 0
    for epoch in range(n_epochs): 
        for j in range(int ( X.shape[1]/ n_batch) ):
            eta = cycleETA(iter)
            j_start = j*n_batch  
            j_end = j_start + n_batch  
            X_batch = X[: , j_start: j_end]
            Y_batch  = Y[: , j_start:j_end] 
            grad_b1 , grad_b2 , grad_W1 , grad_W2 = computeGradAnalytic(X_batch , Y_batch  ,W1,W2, b1 , b2)
            W1 = W1 - eta*grad_W1.reshape((m,d))
            b1 = b1 -eta*grad_b1.reshape((m,1))
            W2 = W2 - eta*grad_W2.reshape((k,m))   
            b2 = b2 -eta*grad_b2.reshape((k,1))          
            activations , probabilities, predictions = evaluateClassifier(X , W1, W2 , b1 , b2)
            cost = computeCost(probabilities, Y, W1 , W2) 
            accuracy = computeAccuracy(predictions , y)
            costValues.append( cost )
            accuracyValues.append(accuracy) 
            # on validation data
            activationsVal, probabilitiesVal, predictionsVal = evaluateClassifier(XVal ,  W1, W2 , b1 , b2)
            costVal = computeCost(probabilitiesVal, YVal, W1 , W2) 
            accuracyVal = computeAccuracy(predictionsVal , yVal)
            costValValues.append(costVal)
            accuracyValValues.append(accuracyVal)
            
            print("Epoch: ", epoch )
            print("Training cost : ", costValues[iter] )
            print("Training Accuracy : ", accuracyValues[iter] ,  "\n ")
            print("Validation cost : ", costValValues[iter])
            print("Validation Accuracy : ", accuracyValValues[iter] ,  "\n ")
            
            iter += 1
    # On test data 
    activationsTest , probabilitiesTest , predictionsTest = evaluateClassifier(XTest, W1, W2 , b1 , b2)
    testAccuracy = computeAccuracy(predictionsTest, yTest)
    print("Test accuracy: ", testAccuracy)
    return accuracyValues  , accuracyValValues , costValues , costValValues ,  W1 , W2 ,  b1 , b2

def plotPerformance(accuracy , accuracyVal , cost  , costVal):
    iterations = list(range(iters))
    plt.figure(1)
    plt.plot(iterations , cost , 'r-' )
    plt.plot(iterations , costVal , 'b-')
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.title("Training and Validation Cost across iteration")
    plt.figure(2)
    plt.plot(iterations, accuracy , 'r-')
    plt.plot(iterations , accuracyVal , 'b-' )    
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy (%)")
    plt.title("Training and Validation Accuracy across iteration")
    plt.show()

def plotWeights(W):
    s_im = np.zeros(k)
    for i in range(k):
        im = W[i , :].reshape(32, 32, 3)
        s_im = (im - np.amin(im) ) / (np.amax(im) - np.amin(im)  )
        plt.imshow(s_im)
        plt.show()

def run():
    #checkGradients()
    W1, W2 , b1, b2 = initParams()  
    accuracyValues  , accuracyValValues , costValues , costValValues ,  W1 , W2 ,  b1 , b2 = miniBatchGradientDescent(W1, W2 , b1 , b2)
    plotPerformance( accuracyValues , accuracyValValues, costValues  , costValValues)
    
if __name__ == '__main__':
    run()
    '''
    To do: 
    - report
    '''