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
l_min = -6
l_max = -3
n_lambda = 10
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
   
def computeGradNumeric(lamda, X, Y, W1, W2 , b1, b2):
    grad_W1 = np.zeros((W1.shape[0] , W1.shape[1]))
    grad_W2 = np.zeros((W2.shape[0] , W2.shape[1]))
    grad_b1 = np.zeros(( m , 1))
    grad_b2 = np.zeros(( k , 1))
    activations, probabilities, predictions = evaluateClassifier(X, W1 , W2 , b1, b2)
    loss ,cost = computeCost(lamda, probabilities, Y ,  W1, W2)
    for i in range(b1.shape[0]):
        b1[i] += h
        activations, probabilities, predictions = evaluateClassifier(X, W1 , W2 , b1, b2)
        loss, cost_try = computeCost(lamda, probabilities, Y, W1, W2)
        grad_b1[i] = (cost_try - cost) / h
        b1[i] -= h
    for i in range(b2.shape[0]):
        b2[i] += h
        activations, probabilities, predictions = evaluateClassifier(X, W1 , W2 , b1, b2)
        loss , cost_try = computeCost(lamda , probabilities, Y, W1, W2)
        grad_b2[i] = (cost_try - cost) / h
        b2[i] -= h
    for i in range(W1.shape[0]):
        for j in range(W1.shape[1]):
            W1[i][j] += h
            activations, probabilities, predictions = evaluateClassifier(X, W1 , W2 , b1, b2)
            loss , cost_try  = computeCost(lamda, probabilities, Y, W1, W2)
            grad_W1[i, j] = (cost_try - cost) / h
            W1[i][j] -= h
    for i in range(W2.shape[0]):
        for j in range(W2.shape[1]):
            W2[i][j] += h
            activations, probabilities, predictions = evaluateClassifier(X, W1 , W2 , b1, b2)
            loss , cost_try = computeCost(lamda, probabilities, Y, W1, W2)
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

def miniBatchGradientDescent(lamda, eta ,  W1, W2  , b1, b2 ):
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
                loss, cost = computeCost(lamda, probabilities, Y, W1 , W2) 
                accuracy = computeAccuracy(predictions , y)
                costValues.append( cost )
                accuracyValues.append(accuracy) 
                lossValues.append(loss)
                # on validation data
                activationsVal, probabilitiesVal, predictionsVal = evaluateClassifier(XVal ,  W1, W2 , b1 , b2)
                lossVal , costVal = computeCost(lamda, probabilitiesVal, YVal, W1 , W2) 
                accuracyVal = computeAccuracy(predictionsVal , yVal)
                costValValues.append(costVal)
                accuracyValValues.append(accuracyVal) 
                lossValValues.append(lossVal)   
                iterations.append(iter)                                              
    # On test data 
    activationsTest , probabilitiesTest , predictionsTest = evaluateClassifier(XTest, W1, W2 , b1 , b2)
    testAccuracy = computeAccuracy(predictionsTest, yTest)
    print("\n" )
    print("Test accuracy: ", testAccuracy, "\n" )
    return  params, iterations, etas, lossValues , lossValValues, accuracyValues  , accuracyValValues , costValues , costValValues , testAccuracy

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
    etaPoints.append(eta_min)
    etaLabels.append("eta_min")
    etaPoints.append(eta_max)
    etaLabels.append("eta_max")
    # plot
    plt.figure(4)
    plt.plot(iters, etas , 'r-') 
    plt.xlabel("Iteration")
    plt.xticks(cyclePoints , cycleLabels)
    plt.ylabel("Learning rate")
    plt.title("Eta values across " + str(n_cycles) +" of iteration with n_s  "+ str(n_s))
    plt.yticks(etaPoints , etaLabels)
    plt.show()

def coarseToFine():
    lamdaValues = list()
    for i in range(n_lambda):
        lamda = cycleLambda()
        lamdaValues.append(lamda)
    filename = "Coarse-to-fine_" + str(n_cycles) + ".txt"
    file = open(filename , "w+" )
    file.write("Results for %i lambda values within the range [ %i , %i ] with batch size of %i :\n\n" %(n_lambda , l_min , l_max , n_batch) )
    for lamda in lamdaValues:
        W1, W2 , b1, b2 = initParams()  
        train = miniBatchGradientDescent(lamda , eta_min ,  W1, W2  , b1, b2 )
        params , iters, etas, lossValues , lossValValues, accuracyValues  , accuracyValValues , costValues , costValValues , testAcc = train
        bestValidResults = np.max(accuracyValValues)
        bestTrainResults = np.max(accuracyValues)
        file.write("Lamda: %f  n_s: %i  total iterations of: %i  total batches of %i  for %i  epochs of training:\n" %(lamda , params[0],params[1] , params[2] , params[3] ) )
        file.write("Best accuracy achieved on training set:  %f\n" %(bestTrainResults)  )
        file.write("Best accuracy achieved on validation set:  %f\n" %(bestValidResults) )
        file.write("Final test accuracy  %f\n\n\n" %(testAcc) )
    file.close()

def run():
    W1, W2 , b1, b2 = initParams()  
    params , iters, etas, lossValues , lossValValues, accuracyValues  , accuracyValValues , costValues , costValValues , testAcc = miniBatchGradientDescent(0.002866 , eta_min, W1, W2 , b1 , b2)
    plotEtas(params[0] , etas)
    plotPerformance( iters, lossValues , lossValValues, accuracyValues , accuracyValValues, costValues  , costValValues)
    
if __name__ == '__main__':
    #checkGradients()
    #coarseToFine()
    run()
    