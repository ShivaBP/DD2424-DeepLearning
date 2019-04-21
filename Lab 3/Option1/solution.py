import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import math
import pickle

d = 3072
k = 10
m = 50
mu = 0
sigma1 =  1 / math.sqrt(d)
sigma2 = 1 / math.sqrt(m)
h = 1e-5
eta_min = 1e-5
eta_max = 1e-1
lamda = 0
n_batch = 100
n_cycles = 3
n_layers = 2

def readData(fileName):
    path = "/Users/shivabp/Desktop/DD2424/Labs/Lab 3/Option1/cifar-10-batches-py/" + fileName
    with open(path, 'rb') as f:
        data = pickle.load(f, encoding='bytes')
    f.close()
    X = (np.array(data[b'data'])).T
    #normalize
    mean_X = np.mean(X )
    std_X = np.std(X )
    X = (X - mean_X) 
    X = X /std_X
    y = np.array(data[b'labels'])
    Y = np.zeros((k, X.shape[1] ))
    for i in range(X.shape[1]):
        Y[y[i]][i] = 1
    return X, Y, y

def init():
    X , Y  , y  = readData("data_batch_1")
    for i in range(2,6):
        filename = "data_batch_" + str(i)
        tempX , tempY , tempy = readData(filename)
        X = np.concatenate((X, tempX) , axis = 1)
        Y = np.concatenate((Y, tempY) , axis = 1)
        y = np.concatenate((y, tempy)) 
    trainX , validX = np.split(X , [49000] , axis = 1 )
    trainY , validY = np.split(Y, [49000] , axis = 1 )
    trainy , validy = np.split(y, [49000]  )
    return trainX , trainY , trainy , validX , validY , validy

def initParams():
    W = list()
    b = list()
    W.append(np.random.normal(mu, sigma1, (m , d)))
    b.append(np.zeros((m , 1)))
    for layer in range(n_layers-1):   
        W.append(np.random.normal(mu, sigma2, ( k ,  m)))
        b.append(np.zeros((k , 1)))
    W = np.array(W)
    b = np.array(b)
    return W , b

def cycleETA(n_s , iter , cycle):
    difference = eta_max - eta_min
    min = 2*cycle*n_s
    middle = (2*cycle + 1)*n_s
    max = 2*(cycle + 1)*n_s
    if (min <=iter  and iter <= middle):
        eta = eta_min + (difference*((iter-min)/n_s))
    elif(middle <= iter  and iter <= max):
        eta = eta_max - (difference*((iter - middle)/n_s))
    return eta

def evaluateClassifier(X , W , b):
    activations = list()
    S = list()
    S.append (np.dot(W[0] , X) + b[0] )
    activations.append(np.maximum(0 , S[0]) ) 
    for layer in range( 1, n_layers ):     
        S.append (np.dot(W[layer] , activations[layer-1]) + b[layer] )
        activations.append(np.maximum(0 , S[layer]) )          
    S = np.array(S)
    activations = np.array(activations)
    final = S[n_layers-1] 
    numerator = np.exp( final )
    probabilities = numerator  / np.sum(numerator , axis =0) 
    predictions = np.argmax(probabilities, axis=0)
    return activations, probabilities , predictions

def computeCost( probabilities,Y ,  W ):
    py = np.multiply(Y, probabilities).sum(axis=0)
    # avoid the error
    py [py  == 0] = np.finfo(float).eps
    weightsSqueredSum = np.zeros()
    for i in range(n_layers):
        weightsSqueredSum  += np.sum(np.square(W[i]) ) 
    l2Reg = lamda * weightsSqueredSum
    loss = ((-np.log(py).sum() / probabilities.shape[1] ))
    cost = loss + l2Reg
    return loss , cost

def computeAccuracy(predictions, y ): 
    totalCorrect = 0
    for i in range(predictions.shape[0]):
        if( predictions[i] ==  y[i] ):
            totalCorrect = totalCorrect + 1
    accuracy = (totalCorrect / predictions.shape[0]) *100
    return accuracy 

def computeGradAnalytic(X , Y, W , b ):
    # lecture 4, slides 30-33
    grad_W = list()
    grad_b = list()
    layer = int (n_layers-1)
    activations , probabilities , predictions = evaluateClassifier(X, W , b)   
    while (layer >= 1):    
        g = - (Y - probabilities)  
        vector = np.ones((X.shape[1] , 1))
        indicator = 1 * (activations[layer-1] > 0)
        grad_b.append(np.dot(g , vector)/ X.shape[1] )
        grad_W.append( (np.dot( g , activations[layer-1].T)/X.shape[1]   ) +(2*lamda*W[layer])  ) 
        g = np.dot(W[layer].T , g)
        g = np.multiply(g, indicator)
        layer = layer -1
    grad_b.append( np.dot(g, vector)/ X.shape[1] )
    grad_W.append( (np.dot( g , X.T)/ X.shape[1] )+ (2*lamda*W[0]) )
    grad_b.reverse()
    grad_W.reverse()
    grad_b = np.array(grad_b)
    grad_W = np.array(grad_W)
    return grad_b , grad_W
   
def computeGradNumeric(X, Y, W1, W2 , b1, b2):
    grad_W1 = np.zeros((W1.shape[0] , W1.shape[1]))
    grad_W2 = np.zeros((W2.shape[0] , W2.shape[1]))
    grad_b1 = np.zeros(( m , 1))
    grad_b2 = np.zeros(( k , 1))
    activations, probabilities, predictions = evaluateClassifier(X, W1 , W2 , b1, b2)
    loss ,cost = computeCost(probabilities, Y ,  W1, W2)
    for i in range(b1.shape[0]):
        b1[i] += h
        activations, probabilities, predictions = evaluateClassifier(X, W1 , W2 , b1, b2)
        loss, cost_try = computeCost(probabilities, Y, W1, W2)
        grad_b1[i] = (cost_try - cost) / h
        b1[i] -= h
    for i in range(b2.shape[0]):
        b2[i] += h
        activations, probabilities, predictions = evaluateClassifier(X, W1 , W2 , b1, b2)
        loss , cost_try = computeCost(probabilities, Y, W1, W2)
        grad_b2[i] = (cost_try - cost) / h
        b2[i] -= h
    for i in range(W1.shape[0]):
        for j in range(W1.shape[1]):
            W1[i][j] += h
            activations, probabilities, predictions = evaluateClassifier(X, W1 , W2 , b1, b2)
            loss , cost_try  = computeCost(probabilities, Y, W1, W2)
            grad_W1[i, j] = (cost_try - cost) / h
            W1[i][j] -= h
    for i in range(W2.shape[0]):
        for j in range(W2.shape[1]):
            W2[i][j] += h
            activations, probabilities, predictions = evaluateClassifier(X, W1 , W2 , b1, b2)
            loss , cost_try = computeCost(probabilities, Y, W1, W2)
            grad_W2[i, j] = (cost_try - cost) / h
            W2[i][j] -= h
    return grad_b1 , grad_b2 , grad_W1 , grad_W2

def checkGradients():
    X, Y, y = readData("data_batch_1")
    W1, W2 , b1 , b2 = initParams()
    grad_b1Analytic , grad_b2Analytic , grad_W1Analytic , grad_W2Analytic  = computeGradAnalytic(0, X[:20 , 0:1], Y[: , 0:1] , W1[: , :20] , W2 , b1, b2 )
    grad_b1Numeric , grad_b2Numeric , grad_W1Numeric , grad_W2Numeric = computeGradNumeric(0, X[:20 , 0:1], Y[: , 0:1] , W1[: , :20] , W2 , b1, b2 )
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

def miniBatchGradientDescent(eta ,  W , b):
    # load data
    X, Y , y , XVal , YVal , yVal = init()
    XTest , YTest , yTest = readData("test_batch")
    #Store results 
    accuracyValues = list()
    accuracyValValues = list()
    costValues = list()
    costValValues = list()
    lossValues = list()
    lossValValues = list()
    iterations =  list()
    etas = list()
    # initialize
    n_s = 2* math.floor(X.shape[1]/ n_batch)
    totIters = int(2* n_cycles*n_s)
    numBatches = int(X.shape[1] /n_batch)
    n_epochs = int (totIters / numBatches)
    params = [n_s, totIters , numBatches , n_epochs]
    iter = 0
    cycleCounter = -1
    for epoch in range(n_epochs): 
        print("Epoch: ", epoch )
        for j in range( numBatches ):
            etas.append(eta)
            j_start = j*n_batch  
            j_end = j_start + n_batch  
            X_batch = X[: , j_start: j_end]
            Y_batch  = Y[: , j_start:j_end] 
            grad_b , grad_W = computeGradAnalytic(X_batch , Y_batch  ,W ,b)
            for layer in range(n_layers):
                W[layer] = W[layer] - (eta * grad_W[layer])
                b[layer] = b[layer] - (eta * grad_b[layer])  
            #update iteration info 
            if (iter % (2 * n_s) == 0):
                cycleCounter +=  1     
            iter += 1       
            eta = cycleETA(n_s, iter , cycleCounter  )         
            # performance check
            # plot results at every 100th point
            '''
            if (iter % 100 == 0): 
                activations , probabilities, predictions = evaluateClassifier(X , W , b)
                loss, cost = computeCost(probabilities, Y, W) 
                accuracy = computeAccuracy(predictions , y)
                costValues.append( cost )
                accuracyValues.append(accuracy) 
                lossValues.append(loss)
                # on validation data
                activationsVal, probabilitiesVal, predictionsVal = evaluateClassifier(XVal ,  W1, W2 , b1 , b2)
                lossVal , costVal = computeCost(probabilitiesVal, YVal, W1 , W2) 
                accuracyVal = computeAccuracy(predictionsVal , yVal)
                costValValues.append(costVal)
                accuracyValValues.append(accuracyVal) 
                lossValValues.append(lossVal)   
                iterations.append(iter)           
            '''                                   
    # On test data 
    activationsTest , probabilitiesTest , predictionsTest = evaluateClassifier(XTest, W, b)
    testAccuracy = computeAccuracy(predictionsTest, yTest)
    print("\n" )
    print("Test accuracy: ", testAccuracy, "\n" )
    return  params, iterations, etas, lossValues , lossValValues, accuracyValues  , accuracyValValues , costValues , costValValues , testAccuracy

def run():
    W , b = initParams()  
    params , iters, etas, lossValues , lossValValues, accuracyValues  , accuracyValValues , costValues , costValValues , testAcc = miniBatchGradientDescent(eta_min, W , b)
    
if __name__ == '__main__':
    run()
    '''
    To do:
    - check gradients
    - fix mini batch
    '''