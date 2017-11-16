import numpy as np

class generalConnection:
    # Stores the # of input nodes, # of output nodes and the functions for the output nodes
    def __init__(self, inNodes, outNodes, functions):
      # Ensures that there is at least 1 node in each layer
      assert(inNodes > 0)
      assert(outNodes > 0)
      self.in_nodes = inNodes
      self.out_nodes = outNodes

      # Creates a random weight array
      self.weight_in_out = np.random.normal(0.0, pow(self.in_nodes,-0.5), (self.out_nodes, self.in_nodes))
      self.functions = functions
      pass

    #Performs the activation function on a given input
    def act(self, x):
      y = self.functions.active(x)
      return y

    #Performs the differential function on a given input
    def diff(self, x):
      y = self.functions.differ(x)
      return y

    #Performs the inverse function on a given input
    def inv(self, x):
      y = self.functions.invert(x)
      return y
