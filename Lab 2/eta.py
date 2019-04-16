import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import math
import pickle
import random

d = 3072
k = 10
m = 50
mu = 0
sigma1 =  1 / math.sqrt(d)
sigma2 = 1 / math.sqrt(m)
h = 1e-5
lamda = 0.002866
eta_max = 1e-1
eta_base = 1.7e-5
n_eta = 10
n_batch = 100
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
    trainX , validX = np.split(X , [49000] , axis = 1 )
    trainY , validY = np.split(Y, [49000] , axis = 1 )
    trainy , validy = np.split(y, [49000]  )
    return trainX , trainY , trainy , validX , validY , validy

def initParams():
    W1 = np.random.normal(mu, sigma1, (m , d))
    W2 = np.random.normal(mu, sigma2, ( k ,  m))
    b1 = np.zeros((m , 1))
    b2 = np.zeros((k , 1))
    return W1, W2 , b1 , b2

def cycleETA( n_s , iter , cycle):
    difference = eta_max - eta_base
    min = 2*cycle*n_s
    middle = (2*cycle + 1)*n_s
    max = 2*(cycle + 1)*n_s
    if (min <=iter  and iter <= middle):
        eta = eta_base + (difference*((iter-min)/n_s))
    elif(middle <= iter  and iter <= max):
        eta = eta_max - (difference*((iter - middle)/n_s))
    return eta

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
    costValues = list() 
    lossValues = list()
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
            grad_b1 , grad_b2 , grad_W1 , grad_W2 = computeGradAnalytic(lamda, X_batch , Y_batch  ,W1,W2, b1 , b2)
            W1 = W1 - (eta * grad_W1)
            b1 = b1 - (eta * grad_b1)
            W2 = W2 - (eta * grad_W2 ) 
            b2 = b2 - (eta * grad_b2)            
            # performance check
            #update iteration info 
            if (iter % (2 * n_s) == 0):
                cycleCounter +=  1     
            iter += 1    
            new =   cycleETA( n_s, iter , cycleCounter  ) 
            eta =    new
            # plot results at every 100th point
            if (iter % 100 == 0): 
                activations , probabilities, predictions = evaluateClassifier(X , W1, W2 , b1 , b2)
                loss, cost = computeCost(lamda, probabilities, Y, W1 , W2) 
                costValues.append( cost )
                lossValues.append(loss)
                iterations.append(iter)                                                      
    # On test data 
    activationsTest , probabilitiesTest , predictionsTest = evaluateClassifier(XTest, W1, W2 , b1 , b2)
    testAccuracy = computeAccuracy(predictionsTest, yTest)
    print("\n" )
    print("Test accuracy: ", testAccuracy, "\n" )
    return  params, iterations, etas, lossValues , costValues , testAccuracy

def plotEtas( n_s , etas): 
    iters = np.arange(len(etas))
    # annotate x axis 
    cyclePoints = np.zeros(int(n_cycles))
    cycleLabels  = list()
    cyclePoints[0] = n_s *2
    cycleLabels.append("2n_s")
    for i in range(1 , len(cyclePoints)):     
        factor =int (2 + int(math.pow(2, i) )) 
        cyclePoints[i] = n_s*factor
        cycleLabels.append( str(factor ) +  "n_s")
    # annotate y axis 
    etaPoints= list()
    etaLabels = list()
    etaPoints.append(eta_base)
    etaLabels.append("eta_min")
    etaPoints.append(eta_max)
    etaLabels.append("eta_max")
    # plot
    plt.figure(1)
    plt.plot(iters, etas , 'r-') 
    plt.xlabel("Iteration")
    plt.xticks(cyclePoints , cycleLabels)
    plt.ylabel("Learning rate")
    plt.title("Eta values across " + str(n_cycles) +" of iteration with n_s  "+ str(n_s))
    plt.yticks(etaPoints , etaLabels)
    plt.show()

def plotPerformance(etas, iters, loss,  cost  ): 
    plt.figure(2)
    plt.plot(iters , cost , 'r-' )
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.title("Training Cost across iterations")
    plt.figure(3)
    plt.plot(iters, loss , 'r-')
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Training Loss across iterations")
    plt.show()

def etaSearch():
    etaMins = list()
    for i in range(n_eta):
        eta_min = random.uniform(0.0000001, 0.001)
        etaMins.append(eta_min )
    filename = "/Users/shivabp/Desktop/DD2424/Labs/Lab 2/Results/bonus2/ETASearch.txt"
    file = open(filename , "w+" )
    for eta_min in etaMins:   
      file.write("Results for %i eta values within the range [ %f , %f ] with batch size of %i :\n\n" %(n_eta , eta_min  , eta_max , n_batch) )
      W1, W2 , b1, b2 = initParams()  
      params , iters, etas, lossValues ,  costValues , testAcc = miniBatchGradientDescent(lamda , eta_min  ,  W1, W2  , b1, b2 )
      file.write("Results for total iterations of: %i  total batches of %i  for %i  epochs of training:\n" %( params[1] , params[2] , params[3] ) )
      file.write("Final test accuracy  %f\n\n\n" %(testAcc) )
    file.close()
    
def run():
    W1, W2 , b1, b2 = initParams()  
    params , iters, etas, lossValues ,  costValues , testAcc = miniBatchGradientDescent(lamda , eta_base, W1, W2 , b1 , b2)
    plotEtas(params[0] , etas)
    plotPerformance( etas, iters, lossValues ,costValues )
    
if __name__ == '__main__':
  etaSearch()
  run()
    