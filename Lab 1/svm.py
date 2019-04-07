import numpy as np
import matplotlib.pyplot as plt
import _pickle as pickle

N = 10000
d = 3072
k = 10
mu = 0
sigma = 0.01
n_batch = 100
n_epochs = 40
delta = 1
eta = 0.01
lamda = 1

def readData(fileName):
    path = "/Users/shivabp/Desktop/DD2424/Labs/Lab 1/cifar-10-batches-py/" + fileName
    with open(path, 'rb') as f:
        data = pickle.load(f, encoding='bytes')
    f.close()
    X = np.array(data[b'data']/255)
    y = np.array(data[b'labels'])
    return X,  y

def initParams():
    W = np.random.normal(mu, sigma, (d, k))
    return W

def classify(X, W):
    scores = np.dot(X, W)
    return scores

def svmLoss(X, W, y):
    # lecture 2
    gradW = np.zeros((W.shape[0] , W.shape[1]))
    loss = 0
    scores = classify(X, W) 
    correctScores = scores[np.arange(X.shape[0]), y].reshape((X.shape[0] , 1))
    margins = np.maximum(0, scores - correctScores[y] + delta)
    margins[np.arange(X.shape[0]), y] = 0
    loss = np.sum(margins) /X.shape[0]
    loss += 0.5 * lamda * np.sum(np.square(W))
    #samples with margin greater than 0 
    # https://math.stackexchange.com/questions/2572318/derivation-of-gradient-of-svm-loss
    temp = np.zeros((X.shape[0] , k))
    temp[margins > 0] = 1
    numTemp = np.sum(temp,axis=1)
    temp[np.arange(X.shape[0]),y] = - numTemp
    gradW = np.dot(X.T , temp )
    gradW /= X.shape[0]
    gradW += lamda*W
    return gradW

def computeAccuracy(scores, y):
    answers= np.argmax(scores, axis =1)
    totalCorrect = 0
    for i in range(scores.shape[0]):
        if( answers[i] ==  y[i] ):
            totalCorrect = totalCorrect + 1
    accuracy = (totalCorrect / scores.shape[0]) *100
    return accuracy 

def miniBatchGradientDescent(W ):
    X,  y = readData("data_batch_1")
    XVal, yVal = readData("data_batch_2")
    XTest,  yTest = readData("test_batch")
    accuracyValues = np.zeros(n_epochs)
    accuracyValValues = np.zeros(n_epochs)
    costValues = np.zeros(n_epochs)
    costValValues = np.zeros(n_epochs)
    for epoch in range(n_epochs):     
        for j in range(int ( X.shape[0] /n_batch) ):
            j_start = j*n_batch  
            j_end = j_start + n_batch
            X_batch = X[ j_start: j_end , :]
            y_batch  = y[ j_start:j_end]    
            wUpdate  = svmLoss(X_batch, W, y_batch)
            W = W - eta * wUpdate
        # on training data
        scores = classify(X, W)
        accuracy = computeAccuracy(scores , y)
        accuracyValues[epoch] = accuracy
        print("Epoch: ", epoch )
        print("Training Accuracy : ", accuracyValues[epoch] ,  "\n ")
        # on validation data
        scoresVal = classify(XVal, W)
        accuracyVal = computeAccuracy(scoresVal , yVal)
        accuracyValValues[epoch] = accuracyVal
        print("Validation Accuracy : ", accuracyValValues[epoch] ,  "\n ")
    # On test data 
    scoresTest = classify(XTest, W)
    testAccuracy = computeAccuracy(scoresTest, yTest)
    print("Test accuracy: ", testAccuracy)
    return accuracyValues , accuracyValValues , testAccuracy 

def plotAcc(accuracy , accuracyVal ):
    epochs = list(range(n_epochs))
    plt.figure(1)
    plt.plot(epochs, accuracy , 'r-')
    plt.plot(epochs , accuracyVal , 'b-' )    
    plt.xlabel("Epoch number")
    plt.ylabel("Accuracy (%)")
    plt.title("Training and Validation Accuracy across epochs")
    plt.show()

def run():
    W = initParams()
    trainingAccuracy , ValidationAccuracy,  testAccuracy =  miniBatchGradientDescent(W )
    plotAcc(trainingAccuracy , ValidationAccuracy)

if __name__ == '__main__':
    run()