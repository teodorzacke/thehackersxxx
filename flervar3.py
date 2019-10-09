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

def f_z(x,y):
    def f2(z):
        return func(x,y,z)
    return opt.fsolve(f2, 0)

def ader(f=f_z, x=0, y=0, h=1e-8, k=1e-4):
    def der(f, x, y, h0=h):
        return ((f(x + h0, y) - f(x, y))/h0), ((f(x, y+h0) - f(x,y))/h0)
    
    xx = (der(f, x+k, y)[0] - der(f, x, y)[0]) / k
    
    yy = (der(f, x, y+k)[1] - der(f, x, y)[1]) / k
    
    yx = (der(f, x+k, y)[1] - der(f, x, y)[1]) / k
    
    X, Y = der(f, x, y, h)
    
    return xx, yy, yx, X, Y
