author = "AI research team"

import ANN
import Datahandler as dh
fileTrain = 'DataTrainSMA.xlsx'
atribut, target = dh.generateToSeries(fileTrain, 3)
hiddenNeoron = 5
print ANN.train(atribut, hiddenNeoron, target, 5)
# print M