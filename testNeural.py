import generalConnectionClass as gc
import generalNetworkClass as gn
import functions as f
import activeFunctionsClass as af

# Testing Code

# Notes:
# Fixed training error, network would use the trained weights to find the error
# of a layer rather than the original weights
# This has been fixed so that the original weights are now used by using a
# variable in the class definition
# Further cleaned up code

# Improvements to make:
# Might create another Function class that uses all elements in a single function
# rather than apply each element to a function
# Require checks for the activeFunctions class with regards to differentiation
# and inverse

# Prints layers within a network
def printNetwork(net):
    print("Layers")
    for cons in net.network:
        print(cons.weight_in_out)
        print("")
        pass
    pass


sigFuncs = af.activeFunction(f.sig, f.diffSig, f.invSig)
reLinFuncs = af.activeFunction(f.reLin, f.diffReLin, f.invReLin)
softPlusFuncs = af.activeFunction(f.softPlus, f.diffSoftPlus,f.invSoftPlus)

conI = gc.generalConnection(3,2, sigFuncs)
conH = gc.generalConnection(2,4, reLinFuncs)
conO = gc.generalConnection(4,2, sigFuncs)

# A network with 2 hidden layers
test = gn.generalNetwork([conI,conH,conO], 0.3)

# A network with no hidden layers
testS = gn.generalNetwork([conI],0.3)

print("Network")
printNetwork(test)
print("")
print("Query", [0.5,0.5,0.5])
print(test.query([0.5,0.5,0.5]))

print("")
print("Train", [0.5,0.5,0.5])
test.train([0.5,0.5,0.5], [0.99,0.01])
printNetwork(test)


print("Single Network")
printNetwork(testS)
print("")
print("Query", [0.5,0.5,0.5])
print(testS.query([0.5,0.5,0.5]))

print("")
print("Train" , [0.5,0.5,0.5])
testS.train([0.5,0.5,0.5], [0.99,0.01])
printNetwork(testS)
