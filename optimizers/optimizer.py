from .ADAM import Adam
from .GradientDescent import SGD

def Optimizer(optim="sgd"):
    if optim == "sgd":
        return SGD
    elif optim == "adam":
        return Adam
    else:
        return None