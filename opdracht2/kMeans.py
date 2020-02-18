import dataProcessing
import numpy as np
import operator
import random
import matplotlib.pyplot as plt
from collections import Counter

"""
returns euclidean distance between two points
"""
def getEuclidean(day, datap):
    return np.linalg.norm(day - datap)

"""
returns eucideans off all points in data set between one point
"""
def getAllEuclideanDistancesOfOnePoint(day, data):
    euclD = [tuple(index, getEuclidean(day, datap)) for index, datap in enumerate(data)]
    sorted_d = sorted(euclD, key=lambda item:item[1])
    return sorted_d

"""
returns euclideans of all points between all points
"""
def getAllEuclideanDistancessOffAllPoints(validation, data):
    euclidians = list()
    for day in validation:
        euclidians.append(getAllEuclideanDistancesOfOnePoint(day, data))
    return euclidians

"""
prints most common label in cluster 
"""
def printMostCommon(clusters, labels):
    for cluster in clusters:
        count = [labels[index] for index in cluster]
        #TODO

"""
get clusters based on centroids
"""
def getClusters(data, centroids):
    clusterList = list()
    #print("centroids", centroids)
    for centroid in centroids:
        clusterList.append(list())
    for key, day in enumerate(data):
        cList = list()
        for centroid in centroids:
            cList.append(getEuclidean(day, centroid))
        clusterList[min(enumerate(cList), key=operator.itemgetter(1))[0]].append(key)
    return clusterList

"""
calculate mean point based on a list of indexes 
"""
def calcMean(data, indexes):
    mean = np.zeros(np.size(data[0]))
    for index in indexes:
        mean += data[index]
    return np.true_divide(mean, len(indexes))

"""
calculate new centroids by calculating the mean point within a cluster
"""
def newCentroids(data, clusters):
    return [calcMean(data, cluster) for cluster in clusters]

"""
recursively determine best cluster, 
first calculate new cluster, compare with old cluster, if it dit not change return it
if it is not the same calculate new centroids and do it all again
"""
def RdetermineBestClusters(data, centroids, cluster):
    newcluster = getClusters(data, centroids)
    if (newcluster == cluster):
        return cluster
    else:
        centroids = newCentroids(data, newcluster)
        return RdetermineBestClusters(data, centroids, newcluster)

"""

"""
def determineBestClusterForK(data, K):
    centroids = [data[random.randint(1, len(data)-1)] for i in range(K)]
    #print(centroids)
    cluster = getClusters(data, centroids)
    centroids = newCentroids(data, cluster)
    return RdetermineBestClusters(data, centroids, cluster)

"""
"""
def calcIntraDistance(centroid, cluster, data):
    distance = 0
    for index in cluster:
        distance += getEuclidean(data[index], centroid)
    return distance / len(cluster)

"""
"""
def getIntraDistanceForClusterForK(data, K):
    centroids = [data[i] for i in random.sample(range(0, len(data)-1), K)]
    clusters = getClusters(data, centroids)
    centroids = newCentroids(data, clusters)
    clusters = RdetermineBestClusters(data, centroids, clusters)
    return [calcIntraDistance(centroid, cluster, data) for centroid, cluster in zip(centroids, clusters)]
   
"""
"""
def printLabelsForCluster(clusters, labels):
    for cluster in clusters:
        print(len(cluster))
        for index in cluster:
            print(labels[index])
        print("\n")

def getIntraDifference(intraDistances):
    return max(intraDistances) - min(intraDistances)

"""
"""
def determineK(data):
    for K in range(2, 5):
        getIntraDifference(getIntraDistanceForClusterForK(data, K))



data = dataProcessing.dataReader('dataset1.csv') 
validation = dataProcessing.dataReader('validation1.csv') 
dates = dataProcessing.datesReader('dataset1.csv')

labels = dataProcessing.addLabels(dates, '2000')
#validationLabels = addLabels(validationDates, '2001')
determineK(data)
#
#printLabelsForCluster(determineBestClusterForK(data, 4), labels)
#printLabelsForCluster(determineBestClusterForK(normalizeData(data, findMinMax(data)), 3), labels)

