import numpy as np

# read data from a file 
# return array with data
def dataReader(csvFile):
    return np.genfromtxt(csvFile, 
        delimiter=';', 
        usecols=[1,2,3,4,5,6,7], 
        converters={5: lambda s: 0 if s == b"-1" else float(s),
            7: lambda s: 0 if s == b"-1" else float(s)});
 
# read data from file
# return only the dates
def datesReader(csvFile):
    return np.genfromtxt(csvFile, delimiter=';', usecols=[0])

# finds the minimum and maximum values of a dataset 
def findMinMax(data):
    foundMinMax = list((list((0,0)),list((0,0)),list((0,0)), list((0,0)),list((0,0)),list((0,0)),list((0,0))))
    for i in range(len(data)):
        for j in range(len(data[i])):
            if data[i][j] < foundMinMax[j][0]:
                foundMinMax[j][0] = data[i][j] 
            elif data[i][j] > foundMinMax[j][1]:
                foundMinMax[j][1] = data[i][j] 
    return foundMinMax

# normalizes the data vased on found minimum and maximum data
def normalizeData(data, foundMinMax):
    for i in range(len(data)):
        for j in range(len(data[i])):
            data[i][j] = ((data[i][j] - foundMinMax[j][0]) / (foundMinMax[j][1] - foundMinMax[j][0])) * 100
            if j == 1 or j == 4:
                data[i][j] = data[i][j] * 5
    return data

# add labels to the dates
def addLabels(dates, year):
    labels = list()
    for date in dates:
      if date < int(year + '0301'):
        labels.append('winter')
      elif int(year + '0301') <= date < int(year + '0601'):
        labels.append('lente')
      elif int(year + '0601') <= date < int(year + '0901'):
        labels.append('zomer')
      elif int(year + '0901') <= date < int(year + '1201'):
        labels.append('herfst')
      else: # from 01-12 to end of year
        labels.append('winter')
    return labels


