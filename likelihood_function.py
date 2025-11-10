#OLS likelihood function

import numpy as np
from scipy.optimize import minimize

# betas are k long
# sigma^2 is our variance, and a single number
# pass in a single array, and then extract the last value to be sigma^2
# going to have k + 1 parameters, k betas, + 1 is going to be sigma^2

def ols_mle(theta, x, y):
    beta = theta[:-1]
    sig2 = theta[-1]
    n = x.shape[0] #number of rows in x

    first = -1 * (n/2) * np.log(2*np.pi)
    second = -1 * (n/2) * np.log(sig2)
    epsilon = y - (x @ beta)
    third = -1*((1/(2*sig2)) * (epsilon.T @ epsilon))

    return -1 * (first + second + third) #negative one is for turning maximum likelihood function to minimize

minimize(ols_mle, [0,0,0], pass in x and y)