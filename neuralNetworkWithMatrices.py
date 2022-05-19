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
        #print("linkLayerMatrix")
        #print(self.linkLayerMatrix)
        
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
        #print("result inputlayer")
        #print(result)
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
        #print("outcome of outputlayer")
        #print(result)
        
                
        def sigmoid(x):
            return 1/(1+math.exp(-x))
        
        #sigmoid(sum(results)) / len(results)
#        test = [sigmoid(np.sum(result[row])) / len(result)  for row in range(result.shape[0])]
        return [sigmoid(np.sum(result[row])) / len(result) for row in range(result.shape[0])]            
    
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
        return self.outputLayer.evaluate()
        
    def setInput(self, data):
        self.inputLayer.setInput(data)

    def getResults(self, testSet):
        results = []
        for label, image in testSet:
            self.setInput(image)
            cross, circle = self.evaluate()
            
            # Take square of the values.
            square = cross**2.0 + circle**2.0
            cross, circle = cross**2.0 / square, circle**2.0 / square
            
            # Inform user of intermediate result.
            stringCrossOfCircle = "cross" if label else "circle"
            print(stringCrossOfCircle + " cross: " + str(cross) + " circle: " + str(circle))
            
            # Process the result.
            if(stringCrossOfCircle == "cross"):
                results.append(cross)
            if(stringCrossOfCircle == "circle"):
                results.append(circle)
        return sum(results) / len(results)
    
class Learner(NeuralNetwork):

    changedLink = 0
    weightChange = 0.1
    newWeigth = 0

        
    def learn(self, testSet):
        
        startRoundCost = 0
        
        for i in range(1000):
            
            steepestLink = 0
            steepestLinkValue = 0
            steepestLinkCosts = 0
            
            startRoundCost = 1 - self.getResults(testSet)
            steepestLinkCosts = startRoundCost
            print("costfactor: " + str(startRoundCost))
            rows, cols = self.inputLayer.linkLayerMatrix.shape
            
            # Change weigth
            for row in range(rows):
                for col in range(cols):
                    oldValue = self.inputLayer.linkLayerMatrix[row][col]
                    newValue = oldValue + self.weightChange * startRoundCost
                    self.inputLayer.linkLayerMatrix[row][col] = newValue
                    costs = 1 - self.getResults(testSet)
                    #print("cost: " + str(costs))
                    # Place the old value back so that every link can be experimented in the same way.
                    self.inputLayer.linkLayerMatrix[row][col] = oldValue
                    
                    # Compare the value of this experiment with the value of the previous most succesfull experiment.
                    # If this experiment is more succesfull, save this experiment as the steepest link.
                    if costs < steepestLinkCosts:
                        steepestLinkCosts = costs
                        steepestLink = (row, col)
                        steepestLinkValue = newValue
                    
            # If this round has made a improvement, then save the improvment and go to the next round.
            # If not, then termintes because improvment is no longer possible.
            if steepestLinkCosts < startRoundCost:
                self.inputLayer.linkLayerMatrix[steepestLink[0], steepestLink[1]] = steepestLinkValue
                print("new value for link " + "row:" + str(row) + "col" + str(col) + " with value: " + str(steepestLinkValue) + "and with the cost of: " + str(steepestLinkCosts))
                print(self.inputLayer.linkLayerMatrix)
            else:
                print("program terminated")
                return
                    
       
    def getResults(self, testSet):
        results = []
        for label, image in testSet:
            self.setInput(image)
            cross, circle = self.evaluate()
            
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
    
    def test(self, testSet):
        for label, image in testSet:
            self.setInput(image)
            cross, circle = self.evaluate()
                
            # Take square of the values.
            square = cross**2.0 + circle**2.0
            cross, circle = cross**2.0 / square, circle**2.0 / square
                
            # Inform user of intermediate result.
            stringCrossOfCircle = "cross" if label else "circle"
            #print(stringCrossOfCircle + " cross: " + str(cross) + " circle: " + str(circle))
                
            result = "cross" if cross > circle else "circle"
            
            print("result: " + str(result) + ", expected result: " +  str(stringCrossOfCircle))

    
learner = Learner()
learner.learn(dataSets.trainSet)
learner.test(dataSets.testSet)
