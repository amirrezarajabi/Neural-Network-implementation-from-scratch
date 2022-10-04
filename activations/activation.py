from .sigmoid import Sigmoid
from .relu import Relu
from .linear import Linear

def Activation(activation_name="sigmoid"):
    if activation_name == "sigmoid":
        return Sigmoid
    elif activation_name == "relu":
        return Relu
    elif activation_name == "linear":
        return Linear
    return None

