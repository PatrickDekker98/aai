import numpy as np
import operator
from collections import Counter

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

def findMinMax(data, foundMinMax):
    for i in range(len(data)):
        for j in range(len(data[i])):
            if data[i][j] < foundMinMax[j][0]:
                foundMinMax[j][0] = data[i][j] 
            elif data[i][j] > foundMinMax[j][1]:
                foundMinMax[j][1] = data[i][j] 
    return foundMinMax

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

# get all euclidean distances of one point
# return a dict with; index : euclidean distance
def getAllEuclideanDistancesOfOnePoint(day, data):
    euclD = list()
    for index, datap in enumerate(data):
        #euclD[index] = np.linalg.norm(day - datap)
        euclD.append(tuple((index, np.linalg.norm(day - datap)))) 
    sorted_d = sorted(euclD, key=lambda item:item[1])
    return sorted_d

def getAllEuclideanDistancessOffAllPoints(validation, data):
    euclidians = list()
    for day in validation:
        euclidians.append(getAllEuclideanDistancesOfOnePoint(day, data))
    return euclidians


# sorts the dict and returns the first Kst items
def daysWithinK(euclideans, K):
    return euclideans[:K]  #dict(itertools.islice(euclideans.items(), K))

# compare a validation label with all data labels in the first Kst items  
# 
def compareLabels(label, dataLabels, withinK):
    count = list()
    for day, val in withinK:
        count.append(labels[day])
        #print(labels[day])
    occurence_count = Counter(count)
    #print(occurence_count)
    if (label ==  occurence_count.most_common(1)[0][0]):
        return 100
    else :
        return 0
    
def deterMainK(dates, validation, dataLabels, validationLabels):
    bestK = 0
    highScore = 0
    allEucledians = getAllEuclideanDistancessOffAllPoints(validation, dates)
    for i in range(1, len(dates)):
        score = 0
        for key, euclediandists in enumerate(allEucledians):
            score += compareLabels(validationLabels[key], dataLabels, euclediandists[:i])
        if (score / len(validation) > highScore):
            highScore = score / len(validation)
            bestK = i
            print(highScore)
            print(bestK)
    return bestK

def printLabel(labels, within):
    count = list()
    for day, val in within:
        count.append(labels[day])
        #print(labels[day])
    occurence_count = Counter(count)
    #print(occurence_count)
    if (occurence_count):
        print(occurence_count.most_common(1)[0][0])
    else :
        print("Nont")
        
def forAllDatesDetermineSeason(days, data, labels, K):
    for day in days:
        deys = daysWithinK(getAllEuclideanDistancesOfOnePoint(day, data), K)
        printLabel(labels, deys)

minMaxList = list((list((0,0)),list((0,0)),list((0,0)), list((0,0)),list((0,0)),list((0,0)),list((0,0))))
data = dataReader('dataset1.csv') 
validation = dataReader('validation1.csv') 
days = dataReader('days.csv')

findMinMax(data, minMaxList)
#normalizeData(data, minMaxList)
#normalizeData(validation, minMaxList)
#normalizeData(days, minMaxList)
dates = datesReader('dataset1.csv')
validationDates = datesReader('validation1.csv') 


labels = addLabels(dates, '2000')

validationLabels = addLabels(validationDates, '2001')

#print(deterMainK(data, validation, labels, validationLabels))

forAllDatesDetermineSeason(days, data, labels, deterMainK(data, validation, labels, validationLabels))
#forAllDatesDetermineSeason(days, data, labels, 64)
