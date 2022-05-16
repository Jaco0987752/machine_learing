# -*- coding: utf-8 -*-
"""
Created on Mon May  9 13:07:46 2022

@author: Jaco
"""

import numpy as np
import dataSets
import math



class Layer:
    def __init__(self, rows, cols, outputs):
        self.rows = rows 
        self.cols = cols
        self.linkLayerMatrix = np.array([ [0.5 for i in range(rows * cols)] for output in range(outputs)])
        print("linkLayerMatrix")
        print(self.linkLayerMatrix)
        
    def evaluate(self):
        raise NotImplementedError()
        
class InputLayer(Layer):
    def __init__(self, rows, cols, outputs):
        super().__init__(rows, cols, outputs)
        self.nodeMatrix = np.array([ 0.5   for i in range(rows * cols)])
        print("input")
        print(self.nodeMatrix)

    def setInput(self, matrix):
        if(len(self.nodeMatrix) == len(matrix)):
            self.nodeMatrix = matrix

    def evaluate(self):
        result = np.multiply(self.nodeMatrix, self.linkLayerMatrix)
        print("result inputlayer")
        print(result)
        return result
        
# This class presents hidden layer.        
class HiddenLayer(Layer):
    def __init__(self, rows, cols, outputs):
        super().__init__(rows, cols, outputs)
        
    def connect(self, previousLayer):
        self.previousLayer = previousLayer
        
    def evaluate(self):
        print("HiddenLayer")
        result = np.multiply(self.previousLayer.evaluate(), self.linkLayerMatrix)
        return result 

# This class represents a output layer.
class OutputLayer(Layer):
    def __init__(self, outputs):
        super().__init__(1, 1, outputs)
        
    def evaluate(self):
        result = self.previousLayer.evaluate()
        print("outcome of outputlayer")
        print(result)
        
                
        def sigmoid(x):
            return 1/(1+math.exp(-x))
        
        #sigmoid(sum(results)) / len(results)
        return [sigmoid(np.sum(result[row])) / len(result)  for row in range(result.shape[0])]
            
    
    def connect(self, inputLayer):
        self.previousLayer = inputLayer        
        
    
# This class represents a network neural network
class NeuralNetwork:
    
    def __init__(self):
        self.inputLayer = InputLayer(3,3,2)
        self.outputLayer = OutputLayer(2)
        #self.hiddenLayers = [HiddenLayer(3,3)]
        
        self.outputLayer.connect(self.inputLayer)
        #self.hiddenLayers[0].connect(self.inputLayer)
        
    def evaluate(self):
        print(self.outputLayer.evaluate())
        
    def setInput(self, data):
        self.inputLayer.setInput(data)
######################################
    def getResults(self, testSet):
        results = []
        for label, image in testSet:
            network.setInput(image)
            cross, circle = network.evaluate()
            
            # Take square of the values.
            square = cross**2.0 + circle**2.0
            cross, circle = cross**2.0 / square, circle**2.0 / square
            
            # Inform user of intermediate result.
            stringCrossOfCircle = "cross" if label else "circle"
            #print(stringCrossOfCircle + " cross: " + str(cross) + " circle: " + str(circle))
            
            # Process the result.
            if(stringCrossOfCircle == "cross"):
                results.append(cross)
            if(stringCrossOfCircle == "circle"):
                results.append(circle)
        return sum(results) / len(results)
    
network = NeuralNetwork()
network.setInput(dataSets.trainSet)
network.evaluate()
