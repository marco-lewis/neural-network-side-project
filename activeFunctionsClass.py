import numpy as np

class activeFunction:
    def __init__(self, func, diff, inv):
        self.function = func
        self.differential = diff
        self.inverse = inv
        pass

    # Maps a function onto each element of a given vector
    def performFuncOnVector(self, func, x):
        y = np.copy(x)
        i = 0
        for v in x:
            y[i] = func(v)
            i += 1
            pass
        return y

    # Performs the active function on a vector for queries
    def active(self, x):
        y = self.performFuncOnVector(self.function, x)
        return y

    # Performs the differential function on a vector for error training
    def differ(self, x):
        y = self.performFuncOnVector(self.differential, x)
        return y

    #Performs the inverse function on a vector when requesting a desired input from a given output
    def invert(self, x):
        y = self.performFuncOnVector(self.inverse, x)
        return y
