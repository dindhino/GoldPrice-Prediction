#Library untuk membentuk MATRIKS pada PYTHON
import numpy as np

#Modul BERAT dan BIAS
#Pada modul ini berat dan bias dirandom pada pertama kali proses
def initWeight(inputSize, num_output):
    w               = 2 * np.random.rand(inputSize, num_output) - 1
    b               = 2 * np.random.rand(1, num_output) - 1
    return w, b

#Modul FUNGSI AKTIVASI untuk propagai MAJU
#Metode yang digunakan adalah sigmoid
def sigmoid_forward(z):
    sig             = 1 / (1 + np.exp(-z))
    return sig

#Modul FUNGSI AKTIVASI untuk propagai MUNDUR
def sigmoid_backward(dout, sig):
    dsig            = np.dot(np.dot(sig, (1 - sig).T), dout)
    return dsig

#Modul untuk menghitung BERAT dikalikan INPUT ditambahkan BIAS pada PROPAGAI MAJU
def affine_forward(x, w, b):
    z               = np.dot(x, w) + b
    # for i in range(x.shape[0]):
    #     z[i,:]=z[i,:]+b
    return z

#Modul untuk menghitung BERAT dikalikan INPUT ditambahkan BIAS pada PROPAGAI MUNDUR
def affine_backward(dout, x, w):
    dx              = np.dot(dout, w.T)
    dw              = np.dot(x.T, dout)
    db              = np.sum(dout, axis=0)
    return dx, dw, db

#Modul kalkulasi ERROR
def errorCalc(output, target):
    error           = [x - y for x, y in zip(target, output)]
    # print np.array(error).shape
    return np.array(error)

#Modul kalkulasi MAPE
def MAPEcalc(output, target):
    # print output
    # selisih = [x - y for x, y in zip(target, output)]
    selisih         = output - target
    # print selisih
    # print target
    return np.mean(np.abs(selisih/target))

#Modul TRAINING
def train(data, num_hidden, target, epoch, lr):
    inputSize       = len(data)
    series          = len(data[0])
    MAPE            = 0
    num_output          = 1
    # x             = np.array(data)
    w0, b0          = initWeight(series, num_hidden)
    w1, b1          = initWeight(num_hidden, num_output)
    print w1.shape, b1.shape, w0.shape, b0.shape
    # print target

    mape = []
    for i in range(epoch):
        mape_ep     = 0
        # print 'epoch=', i
        for d in range(data.shape[0]):

            # print 'fwd 1',
            x               = data[d, :].reshape((1, 3))
            v0              = affine_forward(x, w0, b0)
            a0              = sigmoid_forward(v0)

            # print 'fwd 2'
            v1              = affine_forward(a0, w1, b1)
            a1              = sigmoid_forward(v1)

            # error = errorCalc(a1, target)
            error           = target[d] - a1

            # print 'bck 2'
            da1             = sigmoid_backward(error, a1)
            dout, dw1, db1  = affine_backward(da1, a0, w1)

            # print 'bck 1'
            da0             = sigmoid_backward(dout, a0)
            dout, dw0, db0  = affine_backward(da0, x, w0)

            w1              += lr * dw1
            w0              += lr * dw0
            b1              += lr * db1
            b0              += lr * db0

            mape_ep         += np.abs(error)

        # print mape_ep, data.shape[0]
        mape.append(mape_ep / data.shape[0])
        # MAPE = MAPEcalc(a1, target)
        print 'epoch=', i, (mape_ep / data.shape[0])
        model               = ( w0, b0, w1, b1)
    return mape,model

#Modul TESTING
def test(data,model):
    w0,b0,w1,b1             = model
    y                       =[]
    for d in range(data.shape[0]):

        # print 'fwd 1',
        x                   = data[d, :].reshape((1, 3))
        v0                  = affine_forward(x, w0, b0)
        a0                  = sigmoid_forward(v0)

        # print 'fwd 2'
        v1                  = affine_forward(a0, w1, b1)
        a1                  = sigmoid_forward(v1)
        y.append(a1.tolist()[0][0])

    return y