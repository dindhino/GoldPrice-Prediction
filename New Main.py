#Import semua prosedur dari file Datahandler.py
import Datahandler as dh
import ann2 as ann
import numpy as np

#BAGIAN INI DIKHUSUSKAN UNTUK TRAINING DATA
#CHANGEABLE VARIABLE : File DATA TRAINING
fileTrain       = 'DataTrainSMA.xlsx'
data, target    = dh.generateToSeries(fileTrain, 3)

#CHANGEABLE VARIABLE : HIDDEN NODE
num_hidden      = 5
# print data[1]

#MERUBAH DATA MENJADI MATRIX
data            = np.array(data)
target          = np.array(target)

#Modul TRAINING
#CHANGEABLE VARIABLE : EPOCH dan LEARNING RATE
mape, model     = ann.train(data, num_hidden, target, epoch=1000, lr=0.1)

y               = ann.test(data,model)
print y

mape            = np.mean(np.abs(target-y))
print mape, (1-mape)*100
print 'test'

#BAGIAN INI DIKHUSUSKAN UNTUK TESTING DATA
#CHANGEABLE VARIABLE : File DATA TESTING
fileTest        = 'DataTestSMA.xlsx'
data, target    = dh.generateToSeries(fileTest, 3)
data            = np.array(data)
target          = np.array(target)
y               = ann.test(data,model)

mape            = np.mean(np.abs(target-y))
print mape, (1-mape)*100