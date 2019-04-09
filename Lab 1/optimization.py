import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import _pickle as pickle

N = 10000
d = 3072
k = 10
mu = 0
sigma = 0.01
h = 1e-6
n_batch = 100
n_epochs = 88
lamda = 1
eta = 0.01

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

def initParams():
    W = np.random.normal(mu, sigma, (10, 3072))
    b = np.random.normal(mu, sigma, (10, 1))
    return W, b

def etaDecay(eta):
    eta = eta * 0.9
    return eta

def shuffle(X , Y):
    transposeX = X.T
    transposeY = Y.T
    elementIndices = np.arange(X.shape[1])
    np.random.shuffle(elementIndices)
    shuffledX  =  transposeX[elementIndices]
    shuffledY  =  transposeY[elementIndices]
    X = shuffledX.T
    Y = shuffledY.T
    return X, Y

def choosebestModel(ValidationAccuracy):
    maximumAccuracy = np.max(ValidationAccuracy)
    index = np.argmax(ValidationAccuracy)
    print("The best validation accuracy achieved is:    " , maximumAccuracy , "occured at epoch:    " , index , ".\n")

def evaluateClassifier(X, W, b):
    S = np.dot(W , X) + b
    numerator  = np.exp(S )
    probabilities = numerator / np.sum(numerator , axis = 0)
    predictions = np.argmax(probabilities, axis=0)
    return probabilities, predictions

def computeCost(probabilities, Y, W ):
    py = np.multiply(Y, probabilities).sum(axis=0)
    # avoid the error
    py [py  == 0] = np.finfo(float).eps
    l2Reg = lamda * np.power(W,2).sum()
    return  (-np.log(py).sum() / probabilities.shape[1] + l2Reg )

def computeAccuracy(predictions, y):
    totalCorrect = 0
    for i in range(predictions.shape[0]):
        if( predictions[i] ==  y[i] ):
            totalCorrect = totalCorrect + 1
    accuracy = (totalCorrect / N) *100
    return accuracy 

def computeGradAnalytic(X, W, b,  Y):
    grad_W = np.zeros((W.shape[0] , W.shape[1]))
    grad_b = np.zeros(( k , 1))
    probabilities , predictions = evaluateClassifier(X, W, b)
    vector = np.ones((X.shape[1] , 1 ))
    g = - (Y - probabilities)
    grad_b = np.dot(g, vector )/ X.shape[1]  
    grad_W = np.dot( g , X.T)  / X.shape[1]
    grad_W = np.add (grad_W , (2* lamda* W))
    return grad_W, grad_b

def miniBatchGradientDescent(W, b , eta):
    X, Y, y = readData("data_batch_1")
    XVal, YVal, yVal = readData("data_batch_2")
    XTest, YTest, yTest = readData("test_batch")
    accuracyValues = np.zeros(n_epochs)
    accuracyValValues = np.zeros(n_epochs)
    costValues = np.zeros(n_epochs)
    costValValues = np.zeros(n_epochs)
    for epoch in range(n_epochs):   
        # shuffling 
        #shuffle(X, Y)
        for j in range(int ( X.shape[1] /n_batch) ):
            j_start = j*n_batch  
            j_end = j_start + n_batch
            X_batch = X[: , j_start: j_end]
            Y_batch  = Y[: , j_start:j_end]    
            wUpdate , bUpdate = computeGradAnalytic(X_batch  ,W, b , Y_batch )
            W = W - eta * wUpdate.reshape((k, d))
            b = b - eta * bUpdate.reshape((k, 1)) 
        # on training data
        probabilities, predictions = evaluateClassifier(X , W, b)
        cost = computeCost(probabilities, Y, W) 
        accuracy = computeAccuracy(predictions , y)
        costValues[epoch]  = cost
        accuracyValues[epoch] = accuracy
        print("Epoch: ", epoch )
        print("Training cost : ", costValues[epoch] )
        print("Training Accuracy : ", accuracyValues[epoch] ,  "\n ")
        # on validation data
        probabilitiesVal, predictionsVal = evaluateClassifier(XVal , W, b)
        costVal = computeCost(probabilitiesVal, YVal, W) 
        accuracyVal = computeAccuracy(predictionsVal , yVal)
        costValValues[epoch]  = costVal
        accuracyValValues[epoch] = accuracyVal
        print("Validation cost : ", costValValues[epoch])
        print("Validation Accuracy : ", accuracyValValues[epoch] ,  "\n ")
        # eta decay
        #eta = etaDecay(eta)
    # On test data 
    probabilities, predictionsTest = evaluateClassifier(XTest, W, b)
    testAccuracy = computeAccuracy(predictionsTest, yTest)
    print("Test accuracy: ", testAccuracy)
    return W, costValues,  costValValues, accuracyValues , accuracyValValues , testAccuracy 

def plotCost(accuracy , accuracyVal , cost  , costVal):
    epochs = list(range(n_epochs))
    plt.figure(1)
    plt.plot(epochs , cost , 'r-' )
    plt.plot(epochs, costVal , 'b-')
    plt.xlabel("Epoch number")
    plt.ylabel("Cost")
    plt.title("Training and Validation Cost across epochs")
    plt.figure(2)
    plt.plot(epochs, accuracy , 'r-')
    plt.plot(epochs , accuracyVal , 'b-' )    
    plt.xlabel("Epoch number")
    plt.ylabel("Accuracy (%)")
    plt.title("Training and Validation Accuracy across epochs")
    plt.show()

def run():
    W, b = initParams()
    W,  trainingCost,  validationCost , trainingAccuracy , ValidationAccuracy,  testAccuracy =  miniBatchGradientDescent(W, b , eta)
    #choose the best epoch number for training 
    choosebestModel(ValidationAccuracy)
    plotCost(trainingAccuracy , ValidationAccuracy, trainingCost  , validationCost)

if __name__ == '__main__':
    run()