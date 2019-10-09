#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 10:39:05 2019

@author: teodor
"""

## flervar
import numpy as np
import scipy.optimize as opt
    
def func(x,y,z):
    return x + 2*y + z + np.e ** (2 * z) - 1

# z = z ( x , y )
# x = y = z = 0
# taylor deg 2
# vid x=y=0  för z(x,y)

def f_z(x,y):
    def f2(z):
        return func(x,y,z)
    return opt.fsolve(f2, -4)

def der(f, x, y, h):
    return ((f(x + h, y) - f(x, y))/h)[0], ((f(x, y+h) - f(x,y))/h)[0]


a = der(f_z, 10**-4, 0, 10**-8)[0]
b = der(f_z, 0, 0, 10**-8)[0]
print( (a - b) * 10**4 )

def ader(f, x, y, h1, h2):
    a = (der(f, x+h2, y, h1)[0] - der(f, x, y, h1)[0]) / h2
    b = (der(f, x, y+h2, h1)[1] - der(f, x, y, h1)[1]) / h2
    return a, b