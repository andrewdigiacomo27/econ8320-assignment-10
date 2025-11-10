import numpy as np
from scipy.optimize import minimize


def y(x, z):
    return np.sin(x) - (z + 3)**2

def grad_y(x, z):
    slopeX = (sin(x + 0.05, z) - sin(x, z)) / 0.05
    slopeZ = (sin(x, z + 0.05) - sin(x, z)) / 0.05
    return (slopeX, slopeZ)

def q(xs):
    return -1 * y(xs[0], xs[1])   #doing this to create a maximize and find the optimum

print(q(1.571, -3))
