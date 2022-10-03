from .sigmoid import Sigmoid
from .relu import Relu

def Activation(activation_name="sigmoid"):
    if activation_name == "sigmoid":
        return Sigmoid
    elif activation_name == "relu":
        return Relu
    return None

