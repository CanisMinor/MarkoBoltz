import math
import numpy as np

def pol2cart(theta, r):
    '''
    function: pol2cart
    ------------------
    conversion of polar coordinates to Cartesian coordinates
    :param r: radius of polar coordinate
    :param theta: angle of polar coordinate
    :return: Cartesian coordinates x, y
    '''

    n = len(theta)

    x = np.zeros(n)
    y = np.zeros(n)

    for i in range(0, n):
        x[i] = r * math.cos(theta[i])
        y[i] = r * math.sin(theta[i])

    return x, y

def laplace_func(x, b, c):
    '''
    function: exp_func
    ------------------
    :param x: variable x
    :param b: internal scaling parameter
    :param c: bias
    :return: vector of values c + exp( - x / b ) / b
    '''

    y = np.zeros(len(x))

    for i in range(0, len(x)):
        y[i] = c + math.exp( -x[i] / b) / b

    return y

def exp_func(x, a, b):
    '''
    function: exp_func
    ------------------
    :param x: variable x
    :param a: scaling parameter
    :param b: internal scaling parameter
    :param c: bias
    :return: vector of values a * exp( b * x ) + c
    '''

    y = np.zeros(len(x))

    for i in range(0, len(x)):
        y[i] = a * math.exp(b * x[i]) + 1.0 - a

    return y

