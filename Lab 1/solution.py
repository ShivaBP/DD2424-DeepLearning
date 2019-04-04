import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import _pickle as pickle

N = 10000
d = 3072
k = 10
mu = 0
sigma = 0.01
h = 1e-6
epsilon = 1e-6
lamda = 0.1
n_batch = 100
n_epochs = 40
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
    return  l2Reg , (-np.log(py).sum() / probabilities.shape[1] + l2Reg )

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

def computeGradNumeric(X, W, b,  Y):
    grad_W = np.zeros((W.shape[0], W.shape[1]))
    grad_b = np.zeros((W.shape[0], 1))
    probabilities , predictions = evaluateClassifier(X, W , b)
    l , c  = computeCost(probabilities, Y,  W)
    for i in range (b.shape[0]):
        b_try = np.copy(b)
        b_try[i] = b_try[i] + h
        probabilities_try , predictions = evaluateClassifier(X, W , b_try)
        l , c_try = computeCost(probabilities_try, Y, W )
        grad_b[i] = ( c_try - c ) / h
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            W_try = W
            W_try[i][j] = W_try[i][j] +  h
            probabilities_try , predictions = evaluateClassifier(X, W_try, b)
            l , c_try  = computeCost( probabilities_try, Y, W_try)
            grad_W[i] = (c_try - c)/h
    return grad_W , grad_b

def checkGradients( ):
    X, Y, y = readData("data_batch_1")
    W, b = initParams()
    grad_w_Analytic , grad_b_Analytic  = computeGradAnalytic(X[:5 , 0:2], W[: , :5] , b , Y[: , 0:2]  )
    grad_w_Numeric , grad_b_Numeric = computeGradNumeric(X[:5 , 0:2] , W[: , :5] , b , Y[: , 0:2] )
    print("gradW results:" )
    print('Sum of absolute differences is: ' , np.abs(grad_w_Analytic - grad_w_Numeric).sum())
    print("Analytic gradW:  Mean:   " ,np.abs(grad_w_Analytic).mean() , "   Min:    " ,np.abs(grad_w_Analytic).min() , "    Max:    " ,  np.abs(grad_w_Analytic).max())
    print("Numeric gradW:  Mean:   " ,np.abs(grad_w_Numeric).mean() , "   Min:    " ,np.abs(grad_w_Numeric).min() , "    Max:    " ,  np.abs(grad_w_Numeric).max(), "\n")
    print("gradB results:" )
    print('Sum of absolute differences is: ' , np.abs(grad_b_Analytic - grad_b_Numeric).sum())
    print("Analytic gradb:  Mean:   " ,np.abs(grad_b_Analytic).mean() , "   Min:    " ,np.abs(grad_b_Analytic).min() , "    Max:    " ,  np.abs(grad_b_Analytic).max())
    print("Numeric gradb:  Mean:   " ,np.abs(grad_b_Numeric).mean() , "   Min:    " ,np.abs(grad_b_Numeric).min() , "    Max:    " ,  np.abs(grad_b_Numeric).max(), "\n")

def miniBatchGradientDescent(W, b ):
    X, Y, y = readData("data_batch_1")
    XVal, YVal, YVal = readData("data_batch_2")
    XTest, YTest, yTest = readData("test_batch")
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
            wUpdate , bUpdate = computeGradAnalytic(X_batch  ,W, b , Y_batch )
            W = W - eta * wUpdate.reshape((k, d))
            b = b - eta * bUpdate.reshape((k, 1)) 
        # on training data
        probabilities, predictions = evaluateClassifier(X , W, b)
        loss , cost = computeCost(probabilities, Y, W) 
        costValues[epoch]  = cost
        lossValues[epoch] = loss
        print("Epoch: ", epoch , "\n ")
        print("Training cost : ", costValues[epoch] ,  "\n ")
        # on validation data
        probabilitiesVal, predictions = evaluateClassifier(XVal , W, b)
        lossVal, costVal = computeCost(probabilitiesVal, YVal, W) 
        costValValues[epoch]  = costVal
        lossValValues[epoch] = lossVal
        print("Validation cost : ", costValValues[epoch] ,  "\n ")
    # On test data 
    probabilities, predictionsTest = evaluateClassifier(XTest, W, b)
    testAccuracy = computeAccuracy(predictionsTest, yTest)
    print("Test accuracy: ", testAccuracy)
    return W, costValues,  costValValues, lossValues , lossValValues , testAccuracy 

def plotCost(loss , lossVal , cost  , costVal):
    epochs = list(range(n_epochs))
    plt.figure(1)
    plt.plot(epochs , cost  )
    plt.plot(epochs, costVal )
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

def train():
    W, b = initParams()
    W,  trainingCost,  validationCost , trainingLoss , validationLoss,  testAccuracy =  miniBatchGradientDescent(W, b )
    plotWeights(W)
    plotCost(trainingLoss , validationLoss , trainingCost  , validationCost)

def run():
    checkGradients()
    train()

if __name__ == '__main__':
    run()