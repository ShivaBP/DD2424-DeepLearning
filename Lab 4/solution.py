import numpy as np
import matplotlib.pyplot as plt

k = 0
m = 5
eta = 0.1
seq_length = 25


def init():
    global k
    f = open("/Users/shivabp/Desktop/DD2424/Labs/Lab 4/goblet_book.txt", "r")
    data = list(f.read())
    f.close()
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
    x = x0
    y = []
    for i in range(0, n):
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
    seq = ''
    for i in range(len(y)):
        ind = np.where(y[i] != 0)
        seq += indToChar[ind[0][0]]
    return seq


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
        loss += -np.log(np.dot(Y[:, t], P[t]))
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
        da = dh * (1 - H[t] ** 2)
        dw += np.dot(da, H[t-1].T)
        db += da
        du += np.dot(da, Xt.T)
        du = np.maximum(np.minimum(du, 5), -5)
        dw = np.maximum(np.minimum(dw, 5), -5)
        dv = np.maximum(np.minimum(dv, 5), -5)
        db = np.maximum(np.minimum(db, 5), -5)
        dc = np.maximum(np.minimum(dc, 5), -5)
    # clip the gradioents
    du = np.maximum(np.minimum(du, 5), -5)
    dv = np.maximum(np.minimum(dv, 5), -5)
    dw = np.maximum(np.minimum(dw, 5), -5)
    db = np.maximum(np.minimum(db, 5), -5)
    dc = np.maximum(np.minimum(dc, 5), -5)
    return dw, dv, du, db, dc, H[24]


def computeGradNbumeric(X, Y, b, c, h0,  u, v, w):
    h = 1e-4
    db = np.zeros((b.shape[0], b.shape[1]))
    dc = np.zeros((c.shape[0], c.shape[1]))
    du = np.zeros((u.shape[0], u.shape[1]))
    dw = np.zeros((w.shape[0], w.shape[1]))
    dv = np.zeros((v.shape[0], v.shape[1]))
    for i in range(len(b)):
        safeb = b
        temp = safeb
        temp[i] -= h
        b[i] = temp[i]
        P, H, l1 = evaluateClassifier(X, Y, b, c, h0, u, v, w)
        temp = safeb
        temp[i] += h
        b[i] = temp[i]
        P, H, l2 = evaluateClassifier(X, Y, b, c, h0, u, v, w)
        b = safeb
        db[i] = (l2 - l1) / (h)
    for i in range(len(c)):
        safec = c
        temp = safec
        temp[i] -= h
        c[i] = temp[i]
        P, H, l1 = evaluateClassifier(X, Y, b, c, h0, u, v, w)
        temp = safec
        temp[i] += h
        c[i] = temp[i]
        P, H, l2 = evaluateClassifier(X, Y, b, c, h0, u, v, w)
        c = safec
        dc[i] = (l2 - l1) / (h)
    for i in range(u.shape[0]):
        for j in range(u.shape[1]):
            safeu = u
            temp = safeu
            temp[i][j] -= h
            u[i][j] = temp[i][j]
            P, H, l1 = evaluateClassifier(X, Y, b, c, h0, u, v, w)
            temp = safeu
            temp[i][j] += h
            u[i][j] = temp[i][j]
            P, H, l2 = evaluateClassifier(X, Y, b, c, h0, u, v, w)
            u = safeu
            du[i][j] = (l2 - l1) / (h)
    for i in range(w.shape[0]):
        for j in range(w.shape[1]):
            safew = w
            temp = safew
            temp[i][j] -= h
            w[i][j] = temp[i][j]
            P, H, l1 = evaluateClassifier(X, Y, b, c, h0, u, v, w)
            temp = safew
            temp[i][j] += h
            w[i][j] = temp[i][j]
            P, H, l2 = evaluateClassifier(X, Y, b, c, h0, u, v, w)
            w = safew
            dw[i][j] = (l2 - l1) / (h)
    for i in range(v.shape[0]):
        for j in range(v.shape[1]):
            safev = v
            temp = safev
            temp[i][j] -= h
            v[i][j] = temp[i][j]
            P, H, l1 = evaluateClassifier(X, Y, b, c, h0, u, v, w)
            temp = safev
            temp[i][j] += h
            v[i][j] = temp[i][j]
            P, H, l2 = evaluateClassifier(X, Y, b, c, h0, u, v, w)
            v = safev
            dv[i][j] = (l2 - l1) / (h)
    return dw, dv, du, db, dc


def checkGradients():
    book_data, chars = init()
    charToInd, indToChar = mapContainers(chars)
    h, b, c, u, w, v = initWeights()
    X_chars = book_data[0:seq_length]
    Y_chars = book_data[1:seq_length + 1]
    X = charToOneHot(charToInd, X_chars)
    Y = charToOneHot(charToInd, Y_chars)
    P, H, loss = evaluateClassifier(X, Y, b, c, h, u, v, w)
    dw1, dv1, du1, db1, dc1, H = computeGradAnalytic(P, H, X, Y, b, c, u, v, w)
    dw2, dv2, du2, db2, dc2 = computeGradNbumeric(X, Y, b, c, h,  u, v, w)
    print("gradb results:")
    print('Average of absolute differences is: ',
          np.mean(np.abs(db1 - db2)), "\n")
    print("gradc results:")
    print('Average of absolute differences is: ',
          np.mean(np.abs(dc1 - dc2)), "\n")
    print("gradW results:")
    print('Average of absolute differences is: ',
          np.mean(np.abs(dw1 - dw2)), "\n")
    print("gradU results:")
    print('Average of absolute differences is: ',
          np.mean(np.abs(du1 - du2)), "\n")
    print("gradV results:")
    print('Average of absolute differences is: ',
          np.mean(np.abs(dv1 - dv2)), "\n")


def training():
    book_data, chars = init()
    charToInd, indToChar = mapContainers(chars)
    hprev, b, c, u, w, v = initWeights()
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
    epoch = 0
    smoothLoss = -np.log(1 / k) * seq_length
    smoothLossStore = list()
    while iteration < 100000:
        if (iteration == 0 or e >= len(book_data) - seq_length - 1):
            e = 0
            hprev = np.zeros((m, 1))
            epoch = epoch + 1
        X_chars = book_data[e: e + seq_length]
        Y_chars = book_data[e + 1: e + 1 + seq_length]
        X = charToOneHot(charToInd, X_chars)
        Y = charToOneHot(charToInd, Y_chars)
        P, H, loss = evaluateClassifier(X, Y, b, c, hprev, u, v, w)
        dw, dv, du, db, dc, hprev = computeGradAnalytic(
            P, H, X, Y, b, c,  u, v, w)
        smoothLoss = .999 * smoothLoss + .001 * loss
        smoothLossStore.append(smoothLoss)
        if (iteration % 10000 == 0):
            print("-" * 100)
            print("Synth text iteration " + str(iteration))
            y = synthesis(b, c, hprev, u, v, w, X[:, 0], n=200)
            text = OneHottoChar(y, indToChar)
            print(text)
            print("-" * 70)
        # AdaGrad update
        grads = [dw, dv, du, db, dc]
        weights = [w, v, u, b, c]
        ms = [mW, mV, mU, mb, mc]
        for paramIndex in range(5):
            ms[paramIndex] = ms[paramIndex] + (grads[paramIndex]**2)
            weights[paramIndex] = weights[paramIndex] - \
                ((eta * grads[paramIndex]) / np.sqrt(ms[paramIndex] + 1e-8))
        Ustore.append(u)
        Wstore.append(w)
        Vstore.append(v)
        bstore.append(b)
        cstore.append(c)
        e += seq_length
        iteration += 1
    # Exercise iv)
    bestIndex = np.argmin(smoothLossStore)
    bestU = Ustore[bestIndex]
    bestV = Vstore[bestIndex]
    bestW = Wstore[bestIndex]
    bestb = bstore[bestIndex]
    bestc = cstore[bestIndex]
    print("Passage synthesized by the best model:")
    y = synthesis(bestb, bestc, hprev, bestU, bestV, bestW, X[:, 0], n=1000)
    text = OneHottoChar(y, indToChar)
    print(text)
    print("-" * 100)
    plt.figure(1)
    plt.plot(np.arange(iteration), smoothLossStore)
    plt.xlabel('Iterations')
    plt.ylabel('Smooth Loss')
    plt.show()
    return smoothLossStore


def plot(iters, smoothLoss):
    plt.figure(1)
    plt.plot(iters, smoothLoss)
    plt.xlabel('Iterations')
    plt.ylabel('Smooth Loss')
    plt.show()


if __name__ == '__main__':
    # checkGradients()
    training()
