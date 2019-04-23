import numpy as np
import matplotlib.pyplot as plt
import math
import pickle

d = 3072
k = 10
h = 1e-5
eta_min = 1e-5
eta_max = 1e-1
lamda = 0.005
n_layers = 9

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
    trainX , validX = np.split(X , [45000] , axis = 1 )
    trainY , validY = np.split(Y, [45000] , axis = 1 )
    trainy , validy = np.split(y, [45000]  )
    return trainX , trainY , trainy , validX , validY , validy

def initParams():
    initSigma = 1/np.sqrt(d)
    mu = 0
    # 3layer hidden nodes
    #hiddenNodes = [50 , 50 , k ]
    # 9layer hidden nodes
    hiddenNodes = [50 , 50 , 30 , 20, 20, 10, 10 , 10 ,k  ]
    Ws = [np.random.normal(mu, initSigma, (hiddenNodes[0] , d))]
    bs = [np.zeros((hiddenNodes[0] , 1))]
    for layer in range(1, n_layers):   
        xavierSigma = 1/np.sqrt(Ws[layer-1].shape[0])
        W = np.random.normal(mu, xavierSigma, (hiddenNodes[layer],  Ws[layer-1].shape[0] ))
        b = np.zeros((hiddenNodes[layer]  , 1))
        Ws.append(W)
        bs.append(b)
    return Ws , bs

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
    final = S[n_layers-1] 
    numerator = np.exp( final )
    probabilities = numerator  / np.sum(numerator , axis =0) 
    predictions = np.argmax(probabilities, axis=0)
    return activations, probabilities , predictions

def computeCost( probabilities,Y ,  W ):
    py = np.multiply(Y, probabilities).sum(axis=0)
    # avoid the error
    py [py  == 0] = np.finfo(float).eps
    weightsSqueredSum = 0
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
    g = - (Y - probabilities)  
    # helper
    vector = np.ones((X.shape[1] , 1))
    while (layer >= 1):  
        indicator = 1 * (activations[layer-1] > 0)          
        grad_b.append(np.dot(g , vector)/ X.shape[1] )
        print(g.shape)
        print(activations[layer-1].shape)
        print(W[layer].shape)
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
   
def computeGradNumeric(X, Y, W , b):
    grad_Ws = list()
    grad_bs = list()
    activations, probabilities, predictions = evaluateClassifier(X, W, b)
    for layer in range(n_layers):
        grad_b = np.zeros_like(b[layer])
        grad_W = np.zeros_like(W[layer])
        for i in range(b[layer].shape[0]):
            b[layer][i] += h
            activations, probabilities, predictions = evaluateClassifier(X, W , b)
            loss, cost_try1 = computeCost(probabilities, Y, W)
            b[layer][i] -= h
            activations, probabilities, predictions = evaluateClassifier(X, W , b)
            loss, cost_try2 = computeCost(probabilities, Y, W)
            grad_b[i] = (cost_try1 - cost_try2) / h  
        for i in range(W[layer].shape[0]):
            for j in range(W[layer].shape[1]):
                W[layer][i][j] += h
                activations, probabilities, predictions = evaluateClassifier(X, W , b)
                loss , cost_try1  = computeCost(probabilities, Y, W)
                W[layer][i][j] -= h
                activations, probabilities, predictions = evaluateClassifier(X, W , b)
                loss , cost_try2  = computeCost(probabilities, Y, W)
                grad_W[i][j] =  (cost_try1 - cost_try2) / h   
        grad_bs.append(grad_b)
        grad_Ws.append(grad_W)
    return grad_bs , grad_Ws

def checkGradients():
    X, Y, y = readData("data_batch_1")
    W  , b = initParams()
    # reduce dimensionality for testing 
    X_reduced = X[:20 , 0:1]
    Y_reduced = Y[: , 0:1]
    W_reduced = list()
    W_reduced.append (W[0][: , :20] )
    for layer in range (1, n_layers) :
        W_reduced.append( W[layer])
   # W_reduced = np.asarray((W_reduced))
    grad_bAnalytic  , grad_WAnalytic   = computeGradAnalytic(X_reduced, Y_reduced , W_reduced ,  b )
    grad_bNumeric , grad_WNumeric = computeGradNumeric( X_reduced, Y_reduced , W_reduced ,  b )
    for layer in range (n_layers):
        print("layer: ", layer)
        print("gradW results:" )
        print('Average of absolute differences is: ' , np.mean (np.abs(grad_WAnalytic[layer] - grad_WNumeric[layer])) )
        print("Analytic gradW:  Mean:   " ,np.abs(grad_WAnalytic[layer]).mean() , "   Min:    " ,np.abs(grad_WAnalytic[layer]).min() , "    Max:    " ,  np.abs(grad_WAnalytic[layer]).max())
        print("Numeric gradW:   Mean:   " ,np.abs(grad_WNumeric[layer]).mean() ,  "   Min:    " ,np.abs(grad_WNumeric[layer]).min() ,  "    Max:    " ,  np.abs(grad_WNumeric[layer]).max(), "\n")
        print("gradB results:" )
        print('Average of absolute differences is: ' ,np.mean ( np.abs(grad_bAnalytic[layer] - grad_bNumeric[layer]) ) )
        print("Analytic gradb:  Mean:   " ,np.abs(grad_bAnalytic[layer]).mean() , "   Min:    " ,np.abs(grad_bAnalytic[layer]).min() , "    Max:    " ,  np.abs(grad_bAnalytic[layer]).max())
        print("Numeric gradb:   Mean:   " ,np.abs(grad_bNumeric[layer]).mean() ,  "   Min:    " ,np.abs(grad_bNumeric[layer]).min() ,  "    Max:    " ,  np.abs(grad_bNumeric[layer]).max(), "\n")

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
    # initialize
    n_batch = 100
    n_cycles = 2
    n_s = 5* math.floor(X.shape[1]/ n_batch)
    totIters = int(2* n_cycles*n_s)
    numBatches = int(X.shape[1] /n_batch)
    n_epochs = int (totIters / numBatches)
    iter = 0
    cycleCounter = -1
    for epoch in range(n_epochs): 
        print("Epoch: ", epoch )
        for j in range( numBatches ):
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
            if (iter % 100 == 0): 
                activations , probabilities, predictions = evaluateClassifier(X , W , b)
                loss, cost = computeCost(probabilities, Y, W) 
                accuracy = computeAccuracy(predictions , y)
                costValues.append( cost )
                accuracyValues.append(accuracy) 
                lossValues.append(loss)
                # on validation data
                activationsVal, probabilitiesVal, predictionsVal = evaluateClassifier(XVal ,  W, b)
                lossVal , costVal = computeCost(probabilitiesVal, YVal, W) 
                accuracyVal = computeAccuracy(predictionsVal , yVal)
                costValValues.append(costVal)
                accuracyValValues.append(accuracyVal) 
                lossValValues.append(lossVal)   
                iterations.append(iter)                                                       
    # On test data 
    activationsTest , probabilitiesTest , predictionsTest = evaluateClassifier(XTest, W, b)
    testAccuracy = computeAccuracy(predictionsTest, yTest)
    print("\n" )
    print("Test accuracy: ", testAccuracy, "\n" )
    return iterations, lossValues , lossValValues, accuracyValues  , accuracyValValues , costValues , costValValues

def plotPerformance(iters, loss, lossVal , accuracy , accuracyVal , cost  , costVal): 
    plt.figure(1)
    plt.plot(iters , cost , 'r-' )
    plt.plot(iters, costVal , 'b-')
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.title("Training and Validation Cost across iterations")
    plt.figure(2)
    plt.plot(iters, accuracy , 'r-')
    plt.plot(iters , accuracyVal , 'b-' )    
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy (%)")
    plt.title("Training and Validation Accuracy across iterations")
    plt.figure(3)
    plt.plot(iters, loss , 'r-')
    plt.plot(iters, lossVal , 'b-' )    
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss across iterations")
    plt.show()

def run():
    W , b = initParams()  
    iters,  lossValues , lossValValues, accuracyValues  , accuracyValValues , costValues , costValValues = miniBatchGradientDescent(eta_min, W , b)
    plotPerformance(iters, lossValues, lossValValues , accuracyValues , accuracyValValues , costValues  , costValValues)
    
if __name__ == '__main__':
    checkGradients()
    #run()
    '''
    To do:
    - Exercise 3
    '''