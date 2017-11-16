import math

def oneOr0(x):
    if (x > 0): y = 1
    else: y = 0
    return y

# Define functions in here to use for other usage
# Sigmoid functions
sig = lambda x : 1 / (1 + math.exp(-x))
diffSig = lambda x : sig(x) * (1 - sig(x))
invSig = lambda x : math.log(x/(1-x))

# Rectifier functions
reLin = lambda x : max(0,x)
diffReLin = lambda x : oneOr0(x)
invReLin = lambda x : x

# Softplus functions
softPlus = lambda x : math.log(1 + math.exp(x))
diffSoftPlus = lambda x : sig(x)
invSoftPlus = lambda x : math.log(math.exp(x) - 1)
