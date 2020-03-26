import numpy as np
import neuralnetwork as nn
import csv

def data_reader(csv_file):
    return np.genfromtxt(csv_file, delimiter=',', usecols=[0,1,2,3])

def output_reader(csv_file):
    #return np.genfromtxt(csv_file, dtype=str, delimiter=',', usecols=[4], converters={4: lambda s : {1,0,0} if s ==})
    return np.genfromtxt(csv_file, dtype=str, delimiter=',', usecols=[4], converters={4: lambda s : [(1,0,0)] if s.decode() == 'Iris-setosa' else ([(0,1,0)] if s.decode() == 'Iris-versicolor' else [(0,0,1)])})
    #return np.genfromtxt(csv_file, dtype=str, delimiter=',', usecols=[4], converters={4: lambda s : {1,0,0} if s == "Iris-setosa" else ({0,1,0} if s == "Iris-versicolor" else {0,0,1})})

def findminmax(data):
    foundMinMax = list((list((0,0)),list((0,0)),list((0,0)),list((0,0))))
    for i in range(len(data)):
        for j in range(len(data[i])):
            if data[i][j] < foundMinMax[j][0]:
                foundMinMax[j][0] = data[i][j]
            elif data[i][j] > foundMinMax[j][1]:
                foundMinMax[j][1] = data[i][j]
    return foundMinMax

def normalize_data(data, foundminmax):
    for i in range(len(data)):
        for j in range(len(data[i])):
            data[i][j] = ((data[i][j] - foundminmax[j][0]) / (foundminmax[j][1] - foundminmax[j][0]))
    return data

def validate(neuraln, validation_data, validation_output):
    counter = 0
    for data, val in zip(validation_data, validation_output):
        neuraln.set_input(data)
        print(neuraln.get_output_rounded(), val)
        if np.array_equal(neuraln.get_output_rounded(), val):
            counter += 1
    print("persentage of correct validation: ", (counter / len(validation_data)) * 100, "%" )


data = data_reader("iris.data")
test_outputs = output_reader("iris.data")
validation = data_reader("bezdekIris.data")
validation_outputs = output_reader("bezdekIris.data")

irisnn = nn.neuralnetwork(4, [[0,0,0,0], [0,0,0]])

test_data = normalize_data(data, findminmax(data)) 
validation_data = normalize_data(validation, findminmax(validation))

for dat, out in zip(test_data, test_outputs):
    irisnn.add_train_data(dat.tolist(), out.tolist())

irisnn.train_back_prop()
irisnn.print_all_training_data()

validate(irisnn, validation_data, validation_outputs)
