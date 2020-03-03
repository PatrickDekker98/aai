from neuralnode import *

class layer:
    def __init__(self, nodes = list()):
        self.nodes = nodes
        self.size = len(nodes) 
    
    def return_output(self):
        return [node.get_output() for node in self.nodes]


