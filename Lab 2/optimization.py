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
l_min = -8
l_max = -3
n_batch = 100
n_lambda = 20
n_cycles = 3

def readData(fileName):
    path = "/Users/shivabp/Desktop/DD2424/Labs/Lab 2/cifar-10-batches-py/" + fileName
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
    W1 = np.random.normal(mu, sigma1, (m , d))
    W2 = np.random.normal(mu, sigma2, ( k ,  m))
    b1 = np.zeros((m , 1))
    b2 = np.zeros((k , 1))
    return W1, W2 , b1 , b2

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

def cycleLambda():
    difference = l_max - l_min
    l = l_min + difference* np.random.rand()
    lamda = math.pow(10 , l)
    return lamda

def evaluateClassifier(X , W1, W2 , b1 , b2):
    S1 = np.dot(W1 , X) + b1 
    activations = np.maximum(0 , S1)
    S = np.dot(W2 , activations) + b2  
    numerator = np.exp(S  )
    probabilities = numerator  / np.sum(numerator , axis =0) 
    predictions = np.argmax(probabilities, axis=0)
    return activations, probabilities , predictions

def computeCost(lamda , probabilities,Y ,  W1 , W2 ):
    py = np.multiply(Y, probabilities).sum(axis=0)
    # avoid the error
    py [py  == 0] = np.finfo(float).eps
    l2Reg = lamda * (np.sum( np.square(W1)) + np.sum(np.square(W2)) )
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

def computeGradAnalytic(lamda, X , Y, W1 , W2 , b1, b2 ):
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
   
def miniBatchGradientDescent(lamda, eta ,  W1, W2  , b1, b2 ):
    # load data
    X, Y , y , XVal , YVal , yVal = init()
    XTest , YTest , yTest = readData("test_batch")
    #Store results 
    accuracyValues = list()
    accuracyValValues = list()
    costValues = list()
    costValValues = list()
    iterations =  list()
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
            j_start = j*n_batch  
            j_end = j_start + n_batch  
            X_batch = X[: , j_start: j_end]
            Y_batch  = Y[: , j_start:j_end] 
            grad_b1 , grad_b2 , grad_W1 , grad_W2 = computeGradAnalytic(lamda, X_batch , Y_batch  ,W1,W2, b1 , b2)
            W1 = W1 - (eta * grad_W1)
            b1 = b1 - (eta * grad_b1)
            W2 = W2 - (eta * grad_W2 ) 
            b2 = b2 - (eta * grad_b2)    
            #update iteration info 
            if (iter % (2 * n_s) == 0):
                cycleCounter +=  1     
            iter += 1       
            eta = cycleETA(n_s, iter , cycleCounter  )         
            # performance check
            # plot results at every 100th point
            if (iter % 100 == 0): 
                activations , probabilities, predictions = evaluateClassifier(X , W1, W2 , b1 , b2)
                accuracy = computeAccuracy(predictions , y)
                accuracyValues.append(accuracy) 
                # on validation data
                activationsVal, probabilitiesVal, predictionsVal = evaluateClassifier(XVal ,  W1, W2 , b1 , b2)
                accuracyVal = computeAccuracy(predictionsVal , yVal)
                accuracyValValues.append(accuracyVal) 
                iterations.append(iter)                                              
    # On test data 
    activationsTest , probabilitiesTest , predictionsTest = evaluateClassifier(XTest, W1, W2 , b1 , b2)
    testAccuracy = computeAccuracy(predictionsTest, yTest)
    print("\n" )
    print("Test accuracy: ", testAccuracy, "\n" )
    return  params, iterations,  accuracyValues  , accuracyValValues , testAccuracy

def plotPerformance(iters,  accuracy , accuracyVal ): 
    plt.figure(1)
    plt.plot(iters, accuracy , 'r-')
    plt.plot(iters , accuracyVal , 'b-' )    
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy (%)")
    plt.title("Training and Validation Accuracy across iterations")
    plt.show()

def coarseToFine():
    lamdaValues = list()
    for i in range(n_lambda):
        lamda = cycleLambda()
        lamdaValues.append(lamda)
    path = "/Users/shivabp/Desktop/DD2424/Labs/Lab 2/Results/bonus1_search/Coarse-to-fine" + str(n_lambda) +".txt"
    file = open(path , "w+" )
    file.write("Results for %i lambda values within the range [ %i , %i ] with batch size of %i :\n\n" %(n_lambda , l_min , l_max , n_batch) )
    for lamda in lamdaValues:
        W1, W2 , b1, b2 = initParams()  
        train = miniBatchGradientDescent(lamda , eta_min ,  W1, W2  , b1, b2 )
        params , iters,  accuracyValues  , accuracyValValues ,  testAcc = train
        bestValidResults = np.max(accuracyValValues)
        bestTrainResults = np.max(accuracyValues)
        file.write("Lamda: %f  n_s: %i  total iterations of: %i  total batches of %i  for %i  epochs of training:\n" %(lamda , params[0],params[1] , params[2] , params[3] ) )
        file.write("Best accuracy achieved on training set:  %f\n" %(bestTrainResults)  )
        file.write("Best accuracy achieved on validation set:  %f\n" %(bestValidResults) )
        file.write("Final test accuracy  %f\n\n\n" %(testAcc) )
    file.close()

def run():
    W1, W2 , b1, b2 = initParams()  
    params, iters, accuracyValues  , accuracyValValues , testAcc = miniBatchGradientDescent(0.000432 , eta_min, W1, W2 , b1 , b2)
    plotPerformance( iters, accuracyValues , accuracyValValues )
    
if __name__ == '__main__':
    coarseToFine()
    run()
    