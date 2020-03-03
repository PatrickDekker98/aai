import numpy as np
from layer import * 

class neuron:
    def __init__(self, inputs = layer(), threshold = 0):
        self.inputs = inputs
        self.weights = [0]*len(inputs.nodes)
        self.threshold = threshold
        self.output = 0
        self.learning_rate = 0.1

    def set_threshold(self, threshold):
        self.threshold = threshold

    def get_output(self):
        return int(sum([i * w for i, w in zip(self.inputs.return_output(), self.weights)]) >= self.threshold)

    def update(self, desired_output):
        for i, node in enumerate(self.inputs.nodes):
            delta = self.learning_rate * node.get_output() * (self.get_output() - desired_output)
            self.weights[i] -= delta

    def back_prop(self, desired_output):
        #for inp in self.inputs.nodes:
            
        pass

class input_neuron(neuron):
    def __init__(self, output):
        self.output = output

    def set_threshold(self, threshold):
        pass

    def get_output(self):
        return self.output

    def set_output(self, output):
        self.output = int(output >= 1)

    def update(self, desiredOutput):
        pass
    
    def back_prop(self, desired_output):
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
