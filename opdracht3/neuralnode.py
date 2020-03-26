import numpy as np
from layer import * 
import random 

def sigmoid(x):
    return 1/ (1 + np.exp(-x))

def der_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

def tanh(x):
    return np.tanh(x)

def der_tanh(x):
    return 1.0 - np.tanh(x)**2

class neuron:
    def __init__(self, inputs = layer(), threshold = 0):
        self.inputs = inputs
        self.weights = [random.uniform(-1,1) for i in range(len(inputs.nodes)) ]
        self.threshold = threshold
        self.output = 0
        self.learning_rate = 0.2
        self.weights_out = list()
        self.error_sum = 0

    def set_threshold(self, threshold):
        self.threshold = threshold

    def get_output(self):
        return tanh(sum([i * w for i, w in zip(self.inputs.return_output(), self.weights)]))

    def get_output_round(self):
        return round(self.get_output())

    def update(self, desired_output):
        for i, node in enumerate(self.inputs.nodes):
            self.weights[i] += self.learning_rate * node.get_output() * der_tanh(sum([i * w for i, w in zip(self.inputs.return_output(), self.weights)])) * (desired_output - self.get_output())
            delta = self.learning_rate * node.get_output() * (self.get_output() - desired_output)
            self.weights[i] -= delta

    def back_prop_output(self, desired_output):
        error = der_tanh(sum([i * w for i, w in zip(self.inputs.return_output(), self.weights)])) * (desired_output - self.get_output())
        for i, nw in enumerate(zip(self.inputs.nodes, self.weights)):
            self.weights[i] = nw[1] + self.learning_rate * nw[0].get_output() * error
            nw[0].error_sum += (error * nw[1])
            #nw[0].back_prop(error, nw[1])
            
    def back_prop(self):
        #error = der_tanh(sum([i * w for i, w in zip(self.inputs.return_output(), self.weights)])) * pref_error * pref_weight
        error = der_tanh(sum([i * w for i, w in zip(self.inputs.return_output(), self.weights)])) * self.error_sum 
        for i, nw in enumerate(zip(self.inputs.nodes, self.weights)):
            self.weights[i] = nw[1] + self.learning_rate * nw[0].get_output() * error 
            nw[0].error_sum += (error * nw[1])
            #nw[0].back_prop(error, nw[1])
    
    def reset_error_sum(self):
        self.error_sum = 0
        for inp in self.inputs.nodes:
            inp.reset_error_sum()
    
class input_neuron(neuron):
    def __init__(self, output):
        self.output = output
        self.weights_out = list()
        self.error_sum = 0

    def set_threshold(self, threshold):
        pass

    def get_output(self):
        return self.output

    def set_output(self, output):
        self.output = output 

    def update(self, desiredOutput):
        pass
    
    def back_prop(self ):
        pass
    
    def reset_error_sum(self):
        pass
"""
def norGateF():
    inputNodes = list([node(), node(), node()])
    norGate = node(inputNodes, list([-1, -1, -1]), -0.5)
    print(norGate.getOutput())
    inputNodes[0].setOutput(1)
    print(norGate.getOutput())


def adderF():
    inputNodes = list([node(), node()])
    hiddenLayer = list([node(inputNodes, list([1,1]), 0.5), node(inputNodes, list([-1,-1]), -1.5)])
    sumNode = node(hiddenLayer, list([1,1]), 1.5)
    carryNode = node(inputNodes, list([1,1]), 1.5)
    print(sumNode.getOutput(), carryNode.getOutput())
    inputNodes[0].setOutput(1)
    print(sumNode.getOutput(), carryNode.getOutput())
    inputNodes[0].setOutput(0)
    inputNodes[1].setOutput(1)
    print(sumNode.getOutput(), carryNode.getOutput())
    inputNodes[0].setOutput(1)
    print(sumNode.getOutput(), carryNode.getOutput())

def trainNorGate():
    inputNodes = list([node(), node(), node()])
    norGate = node(inputNodes, list([0, 0, 0]), -0.5)
    norGate.update(1)
    data = [[0,0,1],[0,1,0],[1,0,0],[0,1,1],[1,1,0],[1,0,1],[1,1,1]]
    for dat in data :
        for d, n in zip(dat, inputNodes):
            n.setOutput(d)
            norGate.update(0)
            print(norGate.getOutput())

    inputNodes[0].setOutput(0)
    inputNodes[1].setOutput(0)
    inputNodes[2].setOutput(0)
    norGate.update(1)
    print(norGate.getOutput())


trainNorGate()
"""
