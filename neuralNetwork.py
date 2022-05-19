# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 12:31:20 2022

@author: Jaco
"""
import dataSets
import math

# An abstract class that contains a value of a certain type.
class AbstractContainer:
    def getValue(self):
        raise NotImplementedError()
        
# This class contains the functionality that the nodes types have in common and inherrits from AbstractContainer.
class AbstractNode(AbstractContainer):
    pass

# This class represents a input node.
class InputNode(AbstractNode):
    value = 0
    
    def setValue(self, value):
        self.value = value        
            
    def getValue(self):
        return self.value
 
# This class represents a output node.
class OutputNode(AbstractNode):
    
    def __init__(self, id):
        self.id = id
        self.links = []
        
    def addLink(self, link):
        self.links.append(link)
    
    def getValue(self):
        results = []
        
        def sigmoid(x):
            return 1/(1+math.exp(-x))
        
        for link in self.links:
            results.append((link.inputNode.getValue() * link.getValue()))
        #print("single output:" + str(results))    
        return sigmoid(sum(results)) / len(results)


# This class represents a link between the nodes.
class Link(AbstractContainer):
    linkIdCounter = 0         
    
    def __init__(self, inputNode):
        self.inputNode = inputNode
        self.id = Link.linkIdCounter
        Link.linkIdCounter += 1
        self.value = 0.5
        
    def getValue(self):
        return self.value 
    
    def setValue(self, value):
        self.value = value
        
    
# This class represents a network which is made of inputs and outputs.
# The links connects the inputNodes with the outputNodes.
#
#   inputNode  -  _  
#                 _  -  outputNode
#   outputNode -  _
#                 _  -  outputNode
#   inputNode  - 
#
class Network:
    
    inputNodes = []
    outputNodes = []
    links = []
    
    def __init__(self, inputs, outputs):
        for i in range(inputs):
            self.inputNodes.append(InputNode())
            
        for i in range(outputs):
            self.outputNodes.append(OutputNode(str(i)))
            
        for test_node in self.outputNodes:
            for inputNode in self.inputNodes:

                link = Link(inputNode)
                test_node.addLink(link)
                self.links.append(link)
            print("output node " + test_node.id + " has "+ str(len(test_node.links)) + " links")
                
    def evaluate(self):
        results = []
        for outputNode in self.outputNodes:
            results.append(outputNode.getValue())
        #print("results: " + str(results))
        return results

    def setInput(self, inputs):
        if len(inputs) == len(self.inputNodes):
            for i in range(len(inputs)):
                self.inputNodes[i].setValue(inputs[i])
        else:
            print("list inputs inequal to inputNodes")
    
    def test(self, testSet):
        for label, image in testSet:
            network.setInput(image)
            cross, circle = network.evaluate()
                
            # Take square of the values.
            square = cross**2.0 + circle**2.0
            cross, circle = cross**2.0 / square, circle**2.0 / square
                
            # Inform user of intermediate result.
            stringCrossOfCircle = "cross" if label else "circle"
            #print(stringCrossOfCircle + " cross: " + str(cross) + " circle: " + str(circle))
                
            result = "cross" if cross > circle else "circle"
            
            print("result: " + str(result) + ", expected result: " +  str(stringCrossOfCircle))
    
class Learner(Network):

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
            
            # Change weigth
            for link in self.links:
                oldValue = link.getValue()
                newValue = link.getValue() + self.weightChange * startRoundCost
                link.setValue(newValue)
                costs = 1 - self.getResults(testSet)
                #print("cost: " + str(costs))
                # Place the old value back so that every link can be experimented in the same way.
                link.setValue(oldValue)
                
                # Compare the value of this experiment with the value of the previous most succesfull experiment.
                # If this experiment is more succesfull, save this experiment as the steepest link.
                if costs < steepestLinkCosts:
                    steepestLinkCosts = costs
                    steepestLink = link
                    steepestLinkValue = newValue
                
                    
            # If this round has made a improvement, then save the improvment and go to the next round.
            # If not, then termintes because improvment is no longer possible.
            if steepestLinkCosts < startRoundCost:
                steepestLink.setValue(steepestLinkValue)
                print("new value for link " + str(steepestLink.id) + " with value: " + str(steepestLinkValue) + "and with the cost of: " + str(steepestLinkCosts))
            else:
                print("program terminated")
                return
                    
       
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

    
network = Learner(9,2)
result = network.learn(dataSets.trainSet)

network.test(dataSets.testSet)
