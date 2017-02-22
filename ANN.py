author = "AI research team"

import numpy as np
from operator import truediv
lr = 0.01


def initWeight(inputSize, outputSize):
    # weight = []
    # for i in range(inputSize):
    #     weight.append([[np.random.random() for i in range(series)] for i in range(outputSize)])
    # return np.array(weight)
    w = 2*np.random.rand(inputSize, outputSize)-1
    b = 2*np.random.rand(1, outputSize)-1
    return w, b

def activFunction(l, w, b):
    # print 'l', l.shape
    # print 'w', w.shape
    # print 'b', b.shape
    z = np.dot(l,w)#+b
    #ini digunakan untuk menghirung bias, dikarenakan pada kodingan sebelumnya, bentuk matriks berubah duluan karena perkalian matriks, sehingga bias tidak dapat dikalkulasikan
    for i in range(l.shape[0]):
        z[i,:]=z[i,:]+b
    sig = 1 / (1 + np.exp(-z))
    # print sig.shape
    return sig

def forward(x, w0, w1, b0, b1):
    # l1 =[]
    # output = []
    # for i in range(len(l0)):
    #     l1.append(activFunction(l0[i], w0[i], b0[i]))
    #     output.append(activFunction(l1[i], w1[i], b1[i]))
    # return np.array(l1), np.array(output)
    l0 = activFunction(x,w0,b0)
    l1 = activFunction(l0,w1,b1)
    return l0,l1

def backward(l1,w1,b1,l0,w0,b0,x,error):
    #output dibawah digunakan untuk melakukan tracing dari prosedur ini.
    #print output.shape, w1.shape, l1.shape, b1.shape, w0.shape, l0.shape, b0.shape
    # oks = (1 - output).T
    # print "output transpose = ",oks.shape
    print "error = ",error.shape
    print "l1 = ", l1.shape

    d1 = np.dot(np.dot(l1, (1 - l1).T),error)
    print "d1 = ",d1.shape
    print "l0 = ",l0.shape
    dw1 = lr * np.dot(d1.T,l0)
    print 'dw1',dw1.shape
    db1 = lr * np.sum(d1,axis=0)
    print 'db1',db1.shape

    a0 = np.dot( np.dot(l0, (1 - l0).T),d1  )
    # a0 = np.dot(np.dot(d1, l0.T),(1-l0).T)
    print 'a0',a0.shape
    d0 = np.dot(a0.T, w1)
    print 'd0',d0.shape
    print 'l0',l0.shape
    dw0 = lr * np.dot(d0.T,  l0)
    db0 = lr * np.sum(d0,axis=0)
    print 'dw0',dw0.shape
    w1 = w1 + dw1.T
    b1 = b1 + db1

    w0 = w0 + dw0.T
    b0 = b0 + db0

    return w0, w1, b0, b1

def MAPEcalc(output, target):
    # print output
    # selisih = [x - y for x, y in zip(target, output)]
    selisih = output - target
    # print selisih
    # print target
    return selisih / target

def errorCalc(output, target):
    error = [x - y for x, y in zip(target, output)]
    return np.array(error)

def train(atribut, hidden, target, epoch):
    inputSize = len(atribut)
    series = len(atribut[0])
    MAPE = 0
    outputSize = 1
    x = np.array(atribut)
    w0, b0 = initWeight(series, hidden)
    w1, b1 = initWeight(hidden, outputSize)
    print w1.shape, b1.shape, w0.shape, b0.shape
    # print target

    for i in range(epoch):
        l0, l1 = forward(x, w0, w1, b0, b1)
        # print output
        error = errorCalc(l1, target)
        MAPE = MAPEcalc(l1, target)
        w0, w1, b0, b1 = backward(l1,w1,b1,l0,w0,b0,x,error)
    return MAPE

