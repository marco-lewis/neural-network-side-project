import numpy as np
import generalConnectionClass as gc

class generalNetwork:
    def __init__(self, listOfGeneralConnections, learningRate):
        # The rate at which the network learns at
        self.lr = learningRate
        assert (self.checkConnections(listOfGeneralConnections))
        # The list of general connections within the network
        self.network = listOfGeneralConnections
        # A variable (meant to be used privately)
        # that stores the weights of a layer before it trains that layer
        self.__lastOriginalWeights = listOfGeneralConnections[0]
        pass

    # Checks that the connections within the network are well defined.
    # I.e. # of outputs for one connection = # of inputs for the next connection
    def checkConnections(self,connectionList):
        length = len(connectionList)
        x = 1
        isValid = True
        for con in connectionList:
            if (isValid):
                if (x != 1):
                    isValid = isValid & (con.in_nodes == prev.out_nodes)
                    pass
                x = x + 1
                prev = con
                pass
            pass
        return isValid

    # Train algorithm for the network
    def train(self, inputList, targetList):
        # Code in here follows logic from train connection except without the recursive loop
        # Convert lists into vectors (nx1 arrays)
        inputs = np.array(inputList, ndmin=2).T
        target = np.array(targetList, ndmin=2).T
        layer = 1
        # Trains the first layer
        self.trainConnection(inputs, 0, target)
        pass

    # Trains a single connection in a network given some input, target and the layer
    # within the network the connection currently is
    def trainConnection(self, inputArray, layer, target):
        # What values enter the nodes
        nodeIn = np.dot(self.network[layer].weight_in_out, inputArray)
        # What values leave the nodes
        nodeOut = self.network[layer].act(nodeIn)

        # If not the last connection
        if (layer != len(self.network)-1):
            layer += 1
            # Get the errors from the next layer in the network, as well as train it
            nextError = self.trainConnection(nodeOut, layer, target)
            # Find own error using next weight array in network and next error
            errors = np.dot(self.__lastOriginalWeights.T, nextError)
            # Decrease step so dealing with the current connections own weights now
            layer -= 1
            pass
        else:
            # Get error if last connection
            errors = target - nodeOut
            pass

        # Store the current layer before it is trained
        self.__lastOriginalWeights = self.network[layer].weight_in_out

        # Correct the current weight array
        diff = np.dot((errors * self.network[layer].diff(nodeOut)), np.transpose(inputArray))
        self.network[layer].weight_in_out +=  (self.lr * diff)
        return errors

    # Given an input list, produces a result from the output layer
    # by going through each layer within the network.
    def query(self, inputList):
        values = np.array(inputList, ndmin=2).T
        for nodeConnect in self.network:
            values = np.dot(nodeConnect.weight_in_out, values)
            values = nodeConnect.act(values)
            pass
        return values

    # Given an output list, produces a result from the input layer
    # by going backwards through the network
    def inverseQuery(self, inputList):
        layer = len(self.network) - 1
        while (i >= 0):
            values = self.network[layer].inv(values)
            values = np.dot(self.network[layer].weight_in_out, values)
            layer -= 1
            pass
        return values
