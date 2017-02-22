author = "AI research team $ Anditya Arifianto"

#Library untuk MEMBACA FILE EXCEL
import xlrd
#Library untuk membentuk MATRIKS pada PYTHON
import numpy as np

#Modul untuk MEMBACA File excel lalu memasukkanya kedalam variabel DATA
def readExcel(filename):
    wb              = xlrd.open_workbook(filename)
    data            = wb.sheet_by_index(0)
    data            = [data.cell(i, 0).value for i in range(data.nrows)]
    return data

#Modul untuk melakukan NORMALISASI dari input variabel DATA
def normalisasi(data):
    # rate          = np.mean(data)
    # stdev         = np.std(data)
    # datanorm      = ((data)/stdev )
    min_lama        = min(data)
    max_lama        = max(data)
    min_baru        = 0.1
    max_baru        = 0.9
    data2           = np.array(data)
    # nrm           = np.ones_like(data2)*(min_lama)/(max_lama-min_lama)*(max_baru-min_baru)
    datanorm        = (data2-min_lama)/(max_lama-min_lama)*(max_baru-min_baru)
    return datanorm

#Modul untuk membentuk input berupa TIME SERIES dengan TARGET yang didefinisikan
def generateToSeries(filename, series):
    dataPrice       = readExcel(filename)
    normalizedData  = normalisasi(dataPrice)
    # print dataPrice
    # print normalizedData
    atribut         = []
    target          = []
    for i in range(len(normalizedData)-series):
        atribut.append(normalizedData[i : i+series])
        target.append(normalizedData[i+series])
    return atribut, target

# a,t               = (generateToSeries('DataTrainSMA.xlsx', 3))
# print a
# print t