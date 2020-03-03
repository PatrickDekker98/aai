from neuralnode import *
from layer import *
import random
#from layer import layer, input_layer, output_layer

class neuralnetwork:

    def __init__(self, inputs, shape):
        self.trainingData = list()
        self.layers = list()
        self.layers.append(layer([input_neuron(0) for i in range(inputs)]))
        for i, lay in enumerate(shape):
            self.layers.append(layer([neuron(self.layers[i], tresh) for tresh in lay]))
    
    def print_output(self):
        for output_node in self.layers[len(self.layers) - 1].nodes:
            print(output_node.get_output())
    
    def get_output(self):
        return [output_node.get_output() for output_node in self.layers[len(self.layers) - 1].nodes]

    def set_input(self, inputs):
        for inpune, inpu in zip(self.layers[0].nodes, inputs):
            inpune.set_output(inpu)
    
    def set_weights(self, layer, node, weights):
        self.layers[layer].nodes[node].weights = weights

    def add_train_data(self, inputs, outputs):
        self.trainingData.append([inputs, outputs])

    def check_training(self):
        for trainData in self.trainingData:
            self.set_input(trainData[0])
            if self.get_output() != trainData[1]:
                return False
        return True

    def train_delta_rule(self):
        while not self.check_training():
            for trainData in self.trainingData:
                self.set_input(trainData[0])
                for neuron, i in zip(self.layers[len(self.layers)-1].nodes, trainData[1]):
                    neuron.update(i)

    def train_back_prop(self):
        for trainData in self.trainingData:
            self.set_input(trainData[0])
            for neuron, i in zip(self.layers[len(self.layers)-1].nodes, trainData[1]):
                neuron.back_prop(i) 

def printNorGate():
    norGate = neuralnetwork(3, [[-0.5]])
    norGate.set_weights(1, 0, [-1,-1,-1])
    norGate.print_output()
    norGate.set_input([1,0,0])
    norGate.print_output()

def printAdder():
    addernn = neuralnetwork(2, [[0.5,-1.5,1.5], [1.5,0.5]])
    addernn.set_weights(1,0,[1,1])
    addernn.set_weights(1,1,[-1,-1])
    addernn.set_weights(1,2,[1,1])
    addernn.set_weights(2,0,[1,1,0])
    addernn.set_weights(2,1,[0,0,1])
    addernn.print_output()
    addernn.set_input([1,0])
    addernn.print_output()
    addernn.set_input([0,1])
    addernn.print_output()
    addernn.set_input([1,1])
    addernn.print_output()

def deltaRule():
    norGate = neuralnetwork(3, [[-0.5]])
    norGate.add_train_data([0,0,0],[1])
    norGate.add_train_data([1,0,0],[0])
    norGate.add_train_data([1,1,0],[0])
    norGate.add_train_data([1,1,1],[0])
    norGate.add_train_data([0,1,1],[0])
    norGate.add_train_data([0,0,1],[0])
    norGate.add_train_data([1,0,1],[0])
    norGate.add_train_data([0,1,0],[0])

    norGate.train_delta_rule()

    norGate.set_input([0,0,0])
    norGate.print_output()
    norGate.set_input([1,0,0])
    norGate.print_output()

def backProp():
    xorGate = neuralnetwork(2, [[0.5,-1.5],[1.5]])
    randlist = list()
    xorGate.set_weights(1,0, [ random.uniform(-1,1) for i in range(2) ])
    xorGate.set_weights(1,1, [ random.uniform(-1,1) for i in range(2) ])
    xorGate.set_weights(2,0, [ random.uniform(-1,1) for i in range(2) ])
    xorGate.add_train_data([0,0], [0])
    xorGate.add_train_data([0,1], [1])
    xorGate.add_train_data([1,0], [1])
    xorGate.add_train_data([1,1], [0])
    
    xorGate.set_input([0,0])
    xorGate.print_output()
    xorGate.set_input([0,1])
    xorGate.print_output()
    xorGate.set_input([1,0])
    xorGate.print_output()
    xorGate.set_input([1,1])
    xorGate.print_output()



#printNorGate()
#deltaRule()
backProp()


