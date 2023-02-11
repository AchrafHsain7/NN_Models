import numpy as np

def reLU(Z):
        A = np.maximum(0, Z)
        
        return A
    
def sigmoid(Z):
    A = 1 / (1 + np.exp(-Z))

    return A

def deriv_reLU(dA, activation_cache):
    Z, _ = activation_cache
    dZ = np.array(dA, copy=True)
    dZ[ Z< 0] = 0

    return dZ

def deriv_sigmoid(dA,activation_cache):
    Z,A = activation_cache
    dZ = dA * A*(1-A)

    return dZ