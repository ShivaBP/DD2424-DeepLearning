import numpy as np
import matplotlib.pyplot as plt
import os
import json

k = 0
m = 100
eta = 0.1
seq_length = 20


def init():
    global k
    tweets = list()
    chars = list()
    for jsonfile in os.listdir("/Users/shivabp/Desktop/DD2424/Labs/Lab 4/trump_tweets"):
        with open('/Users/shivabp/Desktop/DD2424/Labs/Lab 4/trump_tweets/%s' % jsonfile) as tweetFile:
            data = json.load(tweetFile)
            for i in range(len(data)):
                tweet = data[i]['text']
                tweets.append(tweet + '<>')  # used as end of tweet char
    for item in tweets:
        for char in item:
            chars.append(char)
    data = list(chars)
    chars = list(set(data))
    k = len(chars)
    return data, chars


def initWeights():
    mu = 0
    sigma = 0.01
    b = np.zeros((m, 1))
    c = np.zeros((k, 1))
    h = np.zeros((m, 1))
    u = np.random.normal(mu, sigma, (m, k))
    w = np.random.normal(mu, sigma, (m, m))
    v = np.random.normal(mu, sigma, (k, m))
    return h, b, c, u, w, v


def mapContainers(chars):
    charToInd = {}
    indToChar = {}
    for i in range(0, len(chars)):
        charToInd[chars[i]] = i
    for i in range(0, len(chars)):
        indToChar[i] = chars[i]
    return charToInd, indToChar


def charToOneHot(charToInd, bookData):
    Y = np.zeros((len(charToInd), len(bookData)))
    for i in range(len(bookData)):
        Y[charToInd[bookData[i]]][i] = 1
    return Y


def synthesis(b, c, h, u, v, w, x0, n):
    y = list()
    x = x0
    for i in range(n):
        a = np.dot(w, h) + np.dot(u, x) + b
        h = np.tanh(a)
        o = np.dot(v, h) + c
        p = np.exp(o) / np.sum(np.exp(o), axis=0)
        cp = np.cumsum(p, axis=0)
        a = np.random.rand()
        ixs = np.nonzero(cp - a > 0)
        ii = ixs[0][0]
        x = np.zeros((k, 1))
        x[ii][0] = 1
        y.append(x)
    return y


def OneHottoChar(y, indToChar):
    sequence = ''
    for i in range(len(y)):
        ind = np.where(y[i] != 0)
        sequence += indToChar[ind[0][0]]
    return sequence


def evaluateClassifier(X, Y, b, c, h, u, v, w):
    P = {}
    H = {}
    H[-1] = h
    loss = 0
    for t in range(X.shape[1]):
        Xt = X[:, t].reshape(X.shape[0], 1)
        at = np.dot(u, Xt) + np.dot(w, H[t-1]) + b
        H[t] = np.tanh(at)
        ot = np.dot(v, H[t]) + c
        P[t] = np.exp(ot) / np.sum(np.exp(ot))
        loss += -np.log(np.dot(Y[:, t].T, P[t]))
    return P, H, loss


def computeGradAnalytic(P, H, X, Y, b, c,  u, v, w):
    db = np.zeros((b.shape[0], b.shape[1]))
    dc = np.zeros((c.shape[0], c.shape[1]))
    du = np.zeros((u.shape[0], u.shape[1]))
    dw = np.zeros((w.shape[0], w.shape[1]))
    dv = np.zeros((v.shape[0], v.shape[1]))
    da = np.zeros((H[0].shape[0], H[0].shape[1]))
    for t in reversed(range(X.shape[1])):
        Yt = Y[:, t].reshape(Y.shape[0], 1)
        Xt = X[:, t].reshape(X.shape[0], 1)
        g = -(Yt - P[t])
        dv += np.dot(g, H[t].T)
        dc += g
        dh = (np.dot(v.T, g) + np.dot(w.T, da))
        da = dh * (1 - (H[t] ** 2))
        dw += np.dot(da, H[t-1].T)
        db += da
        du += np.dot(da, Xt.T)
        # clip the gradioents
        du = np.maximum(np.minimum(du, 5), -5)
        dw = np.maximum(np.minimum(dw, 5), -5)
        dv = np.maximum(np.minimum(dv, 5), -5)
        db = np.maximum(np.minimum(db, 5), -5)
        dc = np.maximum(np.minimum(dc, 5), -5)
    return dw, dv, du, db, dc, H[-1]


def training(maxIter):
    book_data, chars = init()
    charToInd, indToChar = mapContainers(chars)
    h0, b, c, u, w, v = initWeights()
    h = h0
    Ustore = list()
    Vstore = list()
    Wstore = list()
    bstore = list()
    cstore = list()
    mb = np.zeros((b.shape[0], b.shape[1]))
    mc = np.zeros((c.shape[0], c.shape[1]))
    mU = np.zeros((u.shape[0], u.shape[1]))
    mW = np.zeros((w.shape[0], w.shape[1]))
    mV = np.zeros((v.shape[0], v.shape[1]))
    e = 0
    iteration = 0
    smoothLoss = -np.log(1 / k) * seq_length
    smoothLossStore = list()
    while (iteration < maxIter):
        if (iteration == 0 or e >= (len(book_data) - seq_length - 1)):
            e = 0
            h = h0
        X_chars = book_data[e: e + seq_length]
        for X_char in X_chars:
            if (X_char == charToInd.get('<>')):
                h = h0
                break
        Y_chars = book_data[e + 1: e + 1 + seq_length]
        X = charToOneHot(charToInd, X_chars)
        Y = charToOneHot(charToInd, Y_chars)
        P, H, loss = evaluateClassifier(X, Y, b, c, h, u, v, w)
        dw, dv, du, db, dc, hRet = computeGradAnalytic(
            P, H, X, Y, b, c,  u, v, w)
        h = hRet
        smoothLoss = (.999 * smoothLoss) + (.001 * loss)
        smoothLossStore.append(smoothLoss)
        if (iteration % 10000 == 0):
            print("Tweet synthesized at iteration:   " + str(iteration) + "\n")
            y = synthesis(b, c, h, u, v, w, X[:, 0], n=140)
            text = OneHottoChar(y, indToChar)
            print(text, "\n\n")
        # AdaGrad update
        grads = [dw, dv, du, db, dc]
        weights = [w, v, u, b, c]
        ms = [mW, mV, mU, mb, mc]
        for paramIndex in range(5):
            ms[paramIndex] += (grads[paramIndex]**2)
            weights[paramIndex] += - \
                (eta * grads[paramIndex]) / np.sqrt(ms[paramIndex] + 1e-8)
        Ustore.append(u)
        Wstore.append(w)
        Vstore.append(v)
        bstore.append(b)
        cstore.append(c)
        e += seq_length
        iteration += 1
    bestIndex = np.argmin(smoothLossStore)
    bestU = Ustore[bestIndex]
    bestV = Vstore[bestIndex]
    bestW = Wstore[bestIndex]
    bestb = bstore[bestIndex]
    bestc = cstore[bestIndex]
    print("Tweet synthesized by the best model:")
    yBEST = synthesis(bestb, bestc, h, bestU, bestV, bestW, X[:, 0], n=140)
    textBEST = OneHottoChar(yBEST, indToChar)
    print(textBEST)
    return smoothLossStore


if __name__ == '__main__':
    training(300000)
