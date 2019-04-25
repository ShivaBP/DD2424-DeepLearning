import numpy as np
import matplotlib.pyplot as plt
import math
import pickle

d = 3072
k = 10
eta_min = 1e-5
eta_max = 1e-1
lamda = 0.005
n_layers = 4

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
    hiddenNodes = [50  ,  50 ,  30 , k  , 20, 20, 10, 10 , 10 ,k  ]
    Ws = [np.random.normal(mu, initSigma, (hiddenNodes[0] , d))]
    bs = [np.random.normal(mu, initSigma, (hiddenNodes[0] , 1))]
    gammas = [np.random.normal(mu, initSigma, (hiddenNodes[0] , 1))]
    betas = [np.random.normal(mu, initSigma, (hiddenNodes[0] , 1))]
    for layer in range(1, n_layers):   
        xavierSigma = 1/np.sqrt(Ws[layer-1].shape[0])
        W = np.random.normal(mu, xavierSigma, (hiddenNodes[layer],  Ws[layer-1].shape[0] ))
        b = np.random.normal(mu, xavierSigma, (hiddenNodes[layer]  , 1))
        gamma = np.random.normal(mu, xavierSigma, (hiddenNodes[layer]  , 1))
        beta = np.random.normal(mu, xavierSigma, (hiddenNodes[layer]  , 1))
        Ws.append(W)
        bs.append(b)
        gammas.append(gamma)
        betas.append(beta)
    return Ws , bs , gammas , betas

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

def batchNormalize(s , mean = None  ,  var = None ):
  epsilon = 0
  n = s.shape[1]
  if (mean == None):
    mean = np.sum(s/ n)
  if (var == None):
    var = np.sum(np.power((s-mean) ,2 )  , axis= 0  )/ n
  part1 = s - mean
  part2 = np.power(np.diag(var + epsilon ), -0.5)
  sHat = part1 * part2
  return sHat , mean , var

def evaluateClassifier(X , Ws , bs , gammas , betas ):
    activations = [X]
    scores = list()
    means = list()
    variances = list()
    sHats = list()
    sTildes = list()
    for layer in range(  n_layers- 1 ):     
        scores.append (np.dot(Ws[layer] , activations[layer]) + bs[layer] )
        sHat , mean , variance = batchNormalize(scores[layer])
        means.append(mean)
        variances.append(variance)
        sHats.append(sHat)
        sTildes.append( np.multiply(gammas[layer] , sHats[layer])+ bs[layer] )
        activations.append(np.maximum(0 , sTildes[layer]) )       
    final = np.dot(Ws[n_layers -1] , activations[n_layers -1]) + bs[n_layers -1]
    numerator = np.exp( final )
    probabilities = numerator  / np.sum(numerator , axis =0) 
    predictions = np.argmax(probabilities, axis=0)
    return activations, probabilities , predictions , scores ,  sHats , means , variances

def computeCost( probabilities,Y ,  Ws ):
    py = np.multiply(Y, probabilities).sum(axis=0)
    # avoid the error
    py [py  == 0] = np.finfo(float).eps
    weightsSqueredSum = 0
    for i in range(n_layers):
        weightsSqueredSum  += np.sum(np.square(Ws[i]) ) 
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

def batchNormBackPass(g, scores, means , variances):
  # helper
  epsilon = 0
  n=np.array(scores).shape[1]
  vector = np.ones((n , 1))
  sigma1 = np.power((variances + epsilon) , -0.5)
  sigma2 = np.power((variances + epsilon) , -1.5)
  G1=np.multiply(g, np.dot(sigma1, vector.T))
  G2=np.multiply(g, np.dot(sigma2, vector.T))
  D= scores - np.dot(means , vector.T)
  c=np.dot(np.multiply(G2, D) , vector)
  part1 = np.dot(G1 , vector )/n
  part2 = np.multiply(D, np.dot(c , vector.T) )/n
  G = G1 - part1 - part2
  return G

def computeGradAnalytic(X , Y, Ws , bs , gammas , betas ):
  # helper
  n = X.shape[1]
  vector = np.ones((n , 1))
  # lecture 4, slides 30-33
  grad_Ws = list()
  grad_bs = list()
  grad_gammas = list()
  grad_betas = list()
  layer = int (n_layers-1)
  activations , probabilities , predictions ,scores,  sHats , means , variances = evaluateClassifier(X, Ws , bs , gammas , betas)  
  g = - (Y - probabilities)  
  grad_bs.append(np.dot(g , vector)/ n )
  grad_Ws.append( (np.dot( g , activations[layer].T)/n  ) +(2*lamda*Ws[layer])  ) 
  g = np.dot(Ws[layer].T , g)
  indicator = 1 * (activations[layer] > 0) 
  g = np.multiply(g, indicator)
  layer = layer -1 
  while (layer >= 0):  
    grad_gammas.append( np.dot(np.multiply(g, sHats[layer])    , vector  )/ n ) 
    grad_betas.append(np.dot(g , vector )/n )
    g = np.multiply(g, np.dot(gammas[layer] , vector.T))
    g = batchNormBackPass(g, scores[layer], means[layer] , variances[layer])
    grad_bs.append(np.dot(g , vector)/ n )
    grad_Ws.append( (np.dot( g , activations[layer].T)/n  ) +(2*lamda*Ws[layer])  ) 
    if (layer > 0):
        indicator = 1 * (activations[layer] > 0)  
        g = np.dot(Ws[layer].T , g)
        g = np.multiply(g, indicator)
    layer = layer -1
  grad_bs.reverse()
  grad_Ws.reverse()
  grad_gammas.reverse()
  grad_betas.reverse()
  return grad_bs , grad_Ws , grad_gammas , grad_betas
   
def computeGradNumeric(X, Y, W , b , gamma , beta):
  h = 1e-5
  grad_Ws = list()
  grad_bs = list()
  grad_gammas = list()
  grad_betas = list()
  for layer in range(n_layers):
    grad_b = np.zeros_like(b[layer])
    grad_W = np.zeros_like(W[layer])
    grad_gamma = np.zeros_like(gamma[layer])
    grad_beta = np.zeros_like(beta[layer])
    for i in range(b[layer].shape[0]):
      safeb = b[layer]
      temp = safeb
      temp[i] += h
      b[layer] = temp 
      activations, probabilities, predictions ,scores,  sHats , means , variances = evaluateClassifier(X, W , b ,gamma , beta)
      loss, cost_try1 = computeCost(probabilities, Y, W)
      temp = safeb
      temp[i] -= h
      b[layer] = temp 
      activations, probabilities, predictions ,scores,  sHats , means , variances = evaluateClassifier(X, W , b ,gamma , beta)
      loss, cost_try2 = computeCost(probabilities, Y, W)  
      b[layer] = safeb
      grad_b[i] = (cost_try1 - cost_try2) / h  
    for i in range(W[layer].shape[0]):
      for j in range(W[layer].shape[1]):
          safeW = W[layer]
          temp = safeW
          temp[i][j] += h
          W[layer] = temp
          activations, probabilities, predictions , scores, sHats , means , variances = evaluateClassifier(X, W , b ,gamma , beta)
          loss , cost_try1  = computeCost(probabilities, Y, W)
          temp = safeW
          temp[i][j] -= h
          W[layer] = temp
          activations, probabilities, predictions ,scores, sHats , means , variances = evaluateClassifier(X, W , b ,gamma , beta)
          loss , cost_try2  = computeCost(probabilities, Y, W)
          W[layer] = safeW
          grad_W[i][j] =  (cost_try1 - cost_try2) / h        
    for i in range(gamma[layer].shape[0]):
      safeGamma = gamma[layer]
      temp = safeGamma
      temp[i] += h
      gamma[layer]= temp
      activations, probabilities, predictions ,scores, sHats , means , variances = evaluateClassifier(X, W , b ,gamma , beta)
      loss, cost_try1 = computeCost(probabilities, Y, W)
      temp = safeGamma
      temp[i] -= h
      gamma[layer]= temp
      activations, probabilities, predictions ,scores, sHats , means , variances = evaluateClassifier(X, W , b ,gamma , beta)
      loss, cost_try2 = computeCost(probabilities, Y, W)
      gamma[layer] = safeGamma 
      grad_gamma[i] = (cost_try1 - cost_try2) / h
    for i in range(beta[layer].shape[0]):
      safeBeta = beta[layer]
      temp = safeBeta
      temp[i] += h
      beta[layer] = temp
      activations, probabilities, predictions ,scores, sHats , means , variances = evaluateClassifier(X, W , b ,gamma , beta)
      loss, cost_try1 = computeCost(probabilities, Y, W)
      temp = safeBeta
      temp[i] -= h
      beta[layer] = temp
      activations, probabilities, predictions ,scores, sHats , means , variances = evaluateClassifier(X, W , b ,gamma , beta)
      loss, cost_try2 = computeCost(probabilities, Y, W)
      beta[layer] = safeBeta
      grad_beta[i] = (cost_try1 - cost_try2) / h
    grad_bs.append(grad_b)
    grad_Ws.append(grad_W)
    grad_gammas.append(grad_gamma)
    grad_betas.append(grad_beta)
  return grad_bs , grad_Ws , grad_gammas , grad_betas

def checkGradients():
  X, Y, y = readData("data_batch_1")
  Ws , bs , gammas , betas = initParams()
  # reduce dimensionality for testing 
  X_reduced = X[:10 , 0:1]
  Y_reduced = Y[: , 0:1]
  W_reduced = list()
  W_reduced.append (Ws[0][: , :10] )
  for layer in range (1, n_layers) :
    W_reduced.append( Ws[layer])
  grad_bAnalytic  , grad_WAnalytic , grad_gammaAnalytic , grad_betaAnalytic  = computeGradAnalytic(X_reduced, Y_reduced , W_reduced ,  bs , gammas , betas )
  grad_bNumeric , grad_WNumeric , grad_gammaNumeric , grad_betaNumeric = computeGradNumeric( X_reduced, Y_reduced , W_reduced ,  bs , gammas , betas )
  for layer in range (n_layers):
    print("layer: ", layer)
    print("gradW results:" )
    print('Average of absolute differences is: ' , np.mean (np.abs(grad_WAnalytic[layer] - grad_WNumeric[layer])) , "\n")
    print("gradB results:" )
    print('Average of absolute differences is: ' ,np.mean ( np.abs(grad_bAnalytic[layer] - grad_bNumeric[layer]) ) , "\n")
    if ( layer < n_layers -1):
      print("gradGamma results:" )
      print('Average of absolute differences is: ' ,np.mean ( np.abs(grad_gammaAnalytic[layer] - grad_gammaNumeric[layer]) ), "\n" )
      print("gradBeta results:" )
      print('Average of absolute differences is: ' ,np.mean ( np.abs(grad_betaAnalytic[layer] - grad_betaNumeric[layer]) ) , "\n")

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
    Ws , bs , gammas , betas = initParams()
    iters,  lossValues , lossValValues, accuracyValues  , accuracyValValues , costValues , costValValues = miniBatchGradientDescent(eta_min, W , b)
    plotPerformance(iters, lossValues, lossValValues , accuracyValues , accuracyValValues , costValues  , costValValues)
    
if __name__ == '__main__':
    checkGradients()
    #run()