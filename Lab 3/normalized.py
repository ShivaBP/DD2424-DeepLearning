import numpy as np
import matplotlib.pyplot as plt
import math
import pickle

d = 3072
k = 10
eta_min = 1e-5
eta_max = 1e-1
l_min = -6
l_max = -2
n_layers = 3

def readData(fileName):
  path = "/Users/shivabp/Desktop/DD2424/Labs/Lab 3/cifar-10-batches-py/" + fileName
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
  initSigma = np.sqrt(1/d)
  mu = 0
  hiddenNodes = [50 , 50 , k  ]
  sigmas = [1e-1 , 1e-3 , 1e-4]
  Ws = [np.random.normal(mu, initSigma, (hiddenNodes[0] , d))]
  bs = [np.zeros(  (hiddenNodes[0] , 1))]
  gammas = [np.random.normal(mu, initSigma, (hiddenNodes[0] , 1))]
  betas = [np.random.normal(mu, initSigma, (hiddenNodes[0] , 1))]
  for layer in range(1, n_layers):   
    xavierSigma = np.sqrt(1 / Ws[layer-1].shape[0])
    W = np.random.normal(mu, xavierSigma , (hiddenNodes[layer],  Ws[layer-1].shape[0] ))
    b = np.zeros( (hiddenNodes[layer]  , 1))
    gamma = np.random.normal(mu, xavierSigma , (hiddenNodes[layer]  , 1))
    beta = np.random.normal(mu, xavierSigma , (hiddenNodes[layer]  , 1))
    Ws.append(W)
    bs.append(b)
    gammas.append(gamma)
    betas.append(beta)
  return Ws , bs , gammas , betas

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

def exponentialAverage(means , variances):
  alpha = 0.9
  averageMeans = means
  averageVars = variances
  for layer in range(len(means)):
    averageMeans[layer] = (alpha * averageMeans[layer]) + ((1-alpha) * means[layer])
    averageVars[layer] = (alpha * averageVars[layer]) + ((1-alpha) * variances[layer])
  return averageMeans , averageVars

def batchNormalize(s , mean = None  ,  var = None ):
  n = s.shape[1]
  epsilon = 1e-12
  if (mean == None ) :
      mean = np.mean(s, axis=1 , keepdims=True)
  if (var  == None):
      var = np.var(s, axis=1 , keepdims=False)
  part1 =  np.diag(1 / np.sqrt(var+ epsilon))
  part2 = s - mean
  sHat = np.dot(part1 , part2)
  return sHat, mean, var

def evaluateClassifier(X , Ws , bs , gammas , betas ):
  activations = [X]
  scores = list()
  means = list()
  variances = list()
  sHats = list()
  sTildes = list()
  for layer in range(  n_layers- 1 ):     
    scores.append (np.dot(Ws[layer] , activations[layer]) + bs[layer] )
    sHat , mean , variance = batchNormalize(scores[layer] )
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

def computeCost( probabilities,Y ,  Ws , lamda ):
  py = np.multiply(Y, probabilities).sum(axis=0)
  # avoid the error
  py [py  == 0] = np.finfo(float).eps
  weightsSqueredSum = 0
  for i in range(n_layers):
      weightsSqueredSum  = weightsSqueredSum +  np.sum(np.square(Ws[i]) ) 
  l2Reg =  lamda * weightsSqueredSum
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

def batchNormBackPass(g, scores, means ):
  n=scores.shape[1]
  vector = np.ones((n , 1))
  sigma1 = 1 / np.sqrt(np.mean(np.power((scores-means) , 2), axis=1, keepdims=True))
  sigma2 = np.power(sigma1 , 3)
  G1=np.multiply(g, sigma1)
  G2=np.multiply(g, sigma2)
  D = np.subtract(scores , means)
  c = np.sum(np.multiply(G2, D), axis = 1, keepdims = True)
  part1= np.subtract(G1 , np.sum(G1, axis = 1, keepdims = True)/ n  )
  part2 = np.multiply(D, c) /  n 
  G = np.subtract(part1 , part2)
  return G

def computeGradAnalytic(X , Y, Ws , bs , gammas , betas , lamda):
  # helper
  n = X.shape[1]
  vector = np.ones((n , 1))
  # lecture 4, slides 30-33
  grad_Ws = list()
  grad_bs = list()
  grad_gammas = list()
  grad_betas = list()
  layer = int (n_layers-1)
  activations , probabilities , predictions ,scores,  sHats , means , variances = evaluateClassifier(X, Ws , bs , gammas , betas )  
  g = - (Y - probabilities)  
  grad_bs.append(np.dot(g , vector)/ n )
  grad_Ws.append( (np.dot( g , activations[layer].T)/n  ) +(2*lamda*Ws[layer])  ) 
  g = np.dot(Ws[layer].T , g)
  indicator = 1 * (activations[layer] > 0) 
  g = np.multiply(g, indicator)
  layer = layer -1 
  while (layer >= 0):  
    grad_gammas.append( np.dot(np.multiply(g, sHats[layer])  , vector )/ n ) 
    grad_betas.append(np.dot(g , vector )/n )
    g = np.multiply(g, np.dot(gammas[layer] , vector.T))
    g = batchNormBackPass(g, scores[layer], means[layer] )
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
   
def computeGradNumeric(X, Y, W , b , gamma , beta , lamda):
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
      loss, cost_try1 = computeCost(probabilities, Y, W , lamda)
      temp = safeb
      temp[i] -= h
      b[layer] = temp 
      activations, probabilities, predictions ,scores,  sHats , means , variances = evaluateClassifier(X, W , b ,gamma , beta)
      loss, cost_try2 = computeCost(probabilities, Y, W , lamda)  
      b[layer] = safeb
      grad_b[i] = (cost_try1 - cost_try2) / h  
    for i in range(W[layer].shape[0]):
      for j in range(W[layer].shape[1]):
        safeW = W[layer]
        temp = safeW
        temp[i][j] += h
        W[layer] = temp
        activations, probabilities, predictions , scores, sHats , means , variances = evaluateClassifier(X, W , b ,gamma , beta)
        loss , cost_try1  = computeCost(probabilities, Y, W, lamda)
        temp = safeW
        temp[i][j] -= h
        W[layer] = temp
        activations, probabilities, predictions ,scores, sHats , means , variances = evaluateClassifier(X, W , b ,gamma , beta)
        loss , cost_try2  = computeCost(probabilities, Y, W, lamda)
        W[layer] = safeW
        grad_W[i][j] =  (cost_try1 - cost_try2) / h        
    for i in range(gamma[layer].shape[0]):
      safeGamma = gamma[layer]
      temp = safeGamma
      temp[i] += h
      gamma[layer]= temp
      activations, probabilities, predictions ,scores, sHats , means , variances = evaluateClassifier(X, W , b ,gamma , beta)
      loss, cost_try1 = computeCost(probabilities, Y, W, lamda)
      temp = safeGamma
      temp[i] -= h
      gamma[layer]= temp
      activations, probabilities, predictions ,scores, sHats , means , variances = evaluateClassifier(X, W , b ,gamma , beta)
      loss, cost_try2 = computeCost(probabilities, Y, W , lamda)
      gamma[layer] = safeGamma 
      grad_gamma[i] = (cost_try1 - cost_try2) / h
    for i in range(beta[layer].shape[0]):
      safeBeta = beta[layer]
      temp = safeBeta
      temp[i] += h
      beta[layer] = temp
      activations, probabilities, predictions ,scores, sHats , means , variances = evaluateClassifier(X, W , b ,gamma , beta)
      loss, cost_try1 = computeCost(probabilities, Y, W, lamda)
      temp = safeBeta
      temp[i] -= h
      beta[layer] = temp
      activations, probabilities, predictions ,scores, sHats , means , variances = evaluateClassifier(X, W , b ,gamma , beta)
      loss, cost_try2 = computeCost(probabilities, Y, W, lamda)
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
  X_reduced = X[:10 , 0:2]
  Y_reduced = Y[: , 0:2]
  W_reduced = list()
  W_reduced.append (Ws[0][: , :10] )
  for layer in range (1, n_layers) :
    W_reduced.append( Ws[layer])
  grad_bAnalytic  , grad_WAnalytic , grad_gammaAnalytic , grad_betaAnalytic  = computeGradAnalytic(X_reduced, Y_reduced , W_reduced ,  bs , gammas , betas, 0 )
  grad_bNumeric , grad_WNumeric , grad_gammaNumeric , grad_betaNumeric = computeGradNumeric( X_reduced, Y_reduced , W_reduced ,  bs , gammas , betas , 0)
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

def miniBatchGradientDescent(eta ,  W , b , gamma , beta , lamda):
  # load data
  X, Y , y , XVal , YVal , yVal = init()
  XTest , YTest , yTest = readData("test_batch")
  #Store results 
  costValues = list()
  costValValues = list()
  accuracyValValues = list()
  iterations =  list()
  # initialize
  n_batch = 100
  n_cycles = 2
  n_s = 5 * math.floor(X.shape[1]/ n_batch)
  totIters = int(2* n_cycles*n_s)
  numBatches = int(X.shape[1] /n_batch)
  n_epochs = int (totIters / numBatches)
  iter = 0
  cycleCounter = -1
  for epoch in range(n_epochs): 
    print("Epoch: ", epoch )
    X , Y = shuffle(X, Y)
    for j in range( numBatches ):
      j_start = j*n_batch  
      j_end = j_start + n_batch  
      X_batch = X[: , j_start: j_end]
      Y_batch  = Y[: , j_start:j_end] 
      grad_b , grad_W  , grad_gamma , grad_beta= computeGradAnalytic(X_batch , Y_batch  ,W ,b , gamma , beta , lamda)
      for layer in range(n_layers):
        W[layer] = W[layer] - (eta * grad_W[layer])
        b[layer] = b[layer] - (eta * grad_b[layer])   
        if (layer == n_layers-1 ) :
          break
        gamma[layer] = gamma[layer] - (eta * grad_gamma[layer]) 
        beta[layer] = beta[layer] - (eta * grad_beta[layer]) 
      #update iteration info 
      if (iter % (2 * n_s) == 0):
        cycleCounter +=  1     
      iter += 1       
      eta = cycleETA(n_s, iter , cycleCounter  )     
      # performance check
      # plot results at every 100th point
      if (iter % 100 == 0): 
        activations , probabilities, predictions , scores, sHats, means , variances = evaluateClassifier(X , W , b , gamma , beta)
        loss, cost = computeCost(probabilities, Y, W , lamda) 
        costValues.append(cost)
        # on validation data
        activationsVal, probabilitiesVal, predictionsVal , scoresVal , sHatsVal, meansVal , variancesVal = evaluateClassifier(XVal ,  W, b, gamma , beta)
        lossVal , costVal = computeCost(probabilitiesVal, YVal, W, lamda) 
        accuracyVal = computeAccuracy(predictionsVal , yVal)
        costValValues.append(costVal)  
        accuracyValValues.append(accuracyVal) 
        iterations.append(iter)                                                       
  # On test data 
  activationsTest , probabilitiesTest , predictionsTest , scoresTest , sHatsTest , meansTest , variancesTest= evaluateClassifier(XTest, W, b , gamma , beta)
  testAccuracy = computeAccuracy(predictionsTest, yTest)
  print("\n" )
  print("Test accuracy: ", testAccuracy, "\n" )
  return iterations, costValues , costValValues , accuracyValValues , testAccuracy

def coarseToFine():
  n_lambda = 20
  lamdaValues = list()
  for i in range(n_lambda):
    lamda = cycleLambda()
    lamdaValues.append(lamda)
  filename = "Coarse-to-fine_.txt"
  file = open(filename , "w+" )
  file.write("Results for %i lambda values within the range [ %i , %i ]:\n\n" %(n_lambda , l_min , l_max ) )
  for lamda in lamdaValues:
    Ws , bs , gammas , betas = initParams()  
    iters,  costValues , costValValues, accuracyValValues , testAccuracy = miniBatchGradientDescent(eta_min, Ws , bs , gammas , betas , lamda)
    bestValidResults = np.max(accuracyValValues)
    file.write("Lamda: %f  :\n" %(lamda ) )
    file.write("Best accuracy achieved on validation set:  %f\n" %(bestValidResults) )
    file.write("Final test accuracy  %f\n\n\n" %(testAccuracy) )
  file.close()

def plotPerformance(iters, cost  , costVal): 
  plt.figure(1)
  plt.plot(iters , cost , 'r-' )
  plt.plot(iters, costVal , 'b-')
  plt.xlabel("Iteration")
  plt.ylabel("Cost")
  plt.show()

def run():
  Ws , bs , gammas , betas = initParams()
  iters,  costValues , costValValues , accuracyValValues, testAccuracy = miniBatchGradientDescent(eta_min, Ws , bs , gammas , betas , 0.005  )
  plotPerformance(iters,  costValues  , costValValues)
  
if __name__ == '__main__':
  checkGradients()
  coarseToFine()
  run()
  